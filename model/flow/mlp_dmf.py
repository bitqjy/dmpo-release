# MIT License

"""
Decoupled MeanFlow networks for ReinFlow policies.

The DMF paper decouples time conditioning by letting the encoder see the
current time t and the decoder see the next time r.  ReinFlow's MLP policies do
not have transformer blocks to split, so we keep the existing FlowMLP/VisionFlow
input encoder and inject r into the hidden decoder.  The injection is
zero-initialized so a ReFlow checkpoint can be loaded as a warm start and keeps
its original behavior before DMF fine-tuning.
"""

import logging
import math
from copy import deepcopy
from typing import TYPE_CHECKING, List

import torch
import torch.nn as nn
from torch import Tensor

from model.common.mlp import MLP, ResidualMLP
from model.common.modules import RandomShiftsAug, SpatialEmb

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from model.common.vit import VitEncoder


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def _zero_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        nn.init.zeros_(param)
    return module


def _set_requires_grad(module: nn.Module, flag: bool):
    for param in module.parameters():
        param.requires_grad = flag


class _DecoupledHeadMixin:
    """Shared hidden-state r-conditioning for MLP and residual MLP heads."""

    def _init_decoder_conditioning(self, hidden_dim: int, time_dim: int):
        self.r_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.r_to_hidden = _zero_module(nn.Linear(time_dim, hidden_dim))

    def _embed_r(self, r: Tensor, batch_size: int, device: torch.device) -> Tensor:
        if isinstance(r, (int, float)):
            r = torch.ones((batch_size, 1), device=device) * r
        return self.r_embedding(r.view(batch_size, 1)).view(batch_size, self.time_dim)

    def _apply_decoupled_head(self, features: Tensor, r_emb: Tensor) -> Tensor:
        if isinstance(self.mlp_mean, ResidualMLP):
            layers = self.mlp_mean.layers
            x = layers[0](features)
            x = x + self.r_to_hidden(r_emb)
            for layer in layers[1:]:
                x = layer(x)
            return x

        if isinstance(self.mlp_mean, MLP):
            modules = self.mlp_mean.moduleList
            x = modules[0](features)
            x = x + self.r_to_hidden(r_emb)
            for module in modules[1:]:
                x = module(x)
            return x

        raise TypeError(f"Unsupported velocity head type: {type(self.mlp_mean)}")

    def set_encoder_requires_grad(self, flag: bool):
        """Freeze or unfreeze the t-conditioned encoder side."""
        encoder_names = [
            "cond_mlp",
            "time_embedding",
            "backbone",
            "compress",
            "compress1",
            "compress2",
            "prop_emb",
            "cond_embed",
        ]
        for name in encoder_names:
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                _set_requires_grad(module, flag)

        if isinstance(self.mlp_mean, ResidualMLP) and len(self.mlp_mean.layers) > 0:
            _set_requires_grad(self.mlp_mean.layers[0], flag)
        elif isinstance(self.mlp_mean, MLP) and len(self.mlp_mean.moduleList) > 0:
            _set_requires_grad(self.mlp_mean.moduleList[0], flag)

    def set_decoder_requires_grad(self, flag: bool):
        """Freeze or unfreeze the r-conditioned decoder side."""
        _set_requires_grad(self.r_embedding, flag)
        _set_requires_grad(self.r_to_hidden, flag)
        if isinstance(self.mlp_mean, ResidualMLP):
            for layer in self.mlp_mean.layers[1:]:
                _set_requires_grad(layer, flag)
        elif isinstance(self.mlp_mean, MLP):
            for module in self.mlp_mean.moduleList[1:]:
                _set_requires_grad(module, flag)


class DecoupledFlowMLP(nn.Module, _DecoupledHeadMixin):
    """Low-dimensional DMF network compatible with FlowMLP checkpoints."""

    def __init__(
        self,
        horizon_steps,
        action_dim,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.act_dim_total = action_dim * horizon_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.mlp_dims = mlp_dims
        self.activation_type = activation_type
        self.out_activation_type = out_activation_type
        self.use_layernorm = use_layernorm
        self.residual_style = residual_style

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        model = ResidualMLP if residual_style else MLP

        if cond_mlp_dims:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            self.cond_enc_dim = cond_mlp_dims[-1]
        else:
            self.cond_enc_dim = cond_dim

        input_dim = time_dim + action_dim * horizon_steps + self.cond_enc_dim
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [self.act_dim_total],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self._init_decoder_conditioning(hidden_dim=mlp_dims[0], time_dim=time_dim)

    def forward(
        self,
        action,
        time,
        r,
        cond,
        output_embedding=False,
        **kwargs,
    ):
        B, Ta, Da = action.shape
        action_flat = action.view(B, -1)
        state = cond["state"].view(B, -1)
        cond_emb = self.cond_mlp(state) if hasattr(self, "cond_mlp") else state

        if isinstance(time, (int, float)):
            time = torch.ones((B, 1), device=action.device) * time
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)
        r_emb = self._embed_r(r, B, action.device)

        features = torch.cat([action_flat, time_emb, cond_emb], dim=-1)
        vel = self._apply_decoupled_head(features, r_emb)

        if output_embedding:
            return vel.view(B, Ta, Da), time_emb, r_emb, cond_emb
        return vel.view(B, Ta, Da)

    def forward_encoder(self, cond: dict):
        state = cond["state"].view(cond["state"].shape[0], -1)
        return self.cond_mlp(state) if hasattr(self, "cond_mlp") else state

    @torch.no_grad()
    def sample_action(
        self,
        cond: dict,
        inference_steps: int,
        clip_intermediate_actions: bool,
        act_range: List[float],
        z: Tensor = None,
        save_chains: bool = False,
    ):
        B = cond["state"].shape[0]
        device = cond["state"].device
        x_hat = z if z is not None else torch.randn(B, self.horizon_steps, self.action_dim, device=device)
        if save_chains:
            x_chain = torch.zeros((B, inference_steps + 1, self.horizon_steps, self.action_dim), device=device)
            x_chain[:, 0] = x_hat

        t_vals = torch.linspace(0.0, 1.0, inference_steps + 1, device=device)
        for i in range(inference_steps):
            t_curr, r_next = t_vals[i], t_vals[i + 1]
            t = torch.full((B,), t_curr, device=device)
            r = torch.full((B,), r_next, device=device)
            u = self.forward(x_hat, t, r, cond)
            x_hat = x_hat + (r_next - t_curr) * u
            if clip_intermediate_actions or i == inference_steps - 1:
                x_hat = x_hat.clamp(*act_range)
            if save_chains:
                x_chain[:, i + 1] = x_hat

        if save_chains:
            return x_hat, x_chain
        return x_hat


class DecoupledVisionFlowMLP(nn.Module, _DecoupledHeadMixin):
    """Image-conditioned DMF network compatible with VisionFlowMLP checkpoints."""

    def __init__(
        self,
        backbone: "VitEncoder",
        action_dim,
        horizon_steps,
        cond_dim,
        img_cond_steps=1,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        spatial_emb=0,
        visual_feature_dim=128,
        dropout=0,
        num_img=1,
        augment=False,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = action_dim * horizon_steps
        self.prop_dim = cond_dim
        self.img_cond_steps = img_cond_steps
        self.time_dim = time_dim
        self.backbone = backbone
        self.mlp_dims = mlp_dims
        self.activation_type = activation_type
        self.out_activation_type = out_activation_type
        self.use_layernorm = use_layernorm
        self.residual_style = residual_style
        self.spatial_emb = spatial_emb
        self.dropout = dropout
        self.num_img = num_img
        self.augment = augment

        if augment:
            self.aug = RandomShiftsAug(pad=4)
        if spatial_emb > 0:
            assert spatial_emb > 1, "spatial_emb is the embedding dimension and must be > 1"
            if num_img == 2:
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            elif num_img == 1:
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            else:
                raise NotImplementedError(f"num_img={num_img} currently supports only 1 or 2")
            visual_feature_dim = spatial_emb * num_img
        else:
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
        self.cond_enc_dim = visual_feature_dim + self.prop_dim

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        input_dim = time_dim + action_dim * horizon_steps + self.cond_enc_dim
        output_dim = action_dim * horizon_steps
        model = ResidualMLP if residual_style else MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self._init_decoder_conditioning(hidden_dim=mlp_dims[0], time_dim=time_dim)

    def _encode_cond(self, cond: dict) -> Tensor:
        import einops

        B = cond["state"].shape[0]
        _, T_rgb, C, H, W = cond["rgb"].shape
        state = cond["state"].view(B, -1)
        rgb = cond["rgb"][:, -self.img_cond_steps :]
        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        elif self.num_img == 1:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        else:
            raise ValueError(f"self.num_img={self.num_img} < 1")
        rgb = rgb.float()

        if self.num_img == 2:
            rgb1, rgb2 = rgb[:, 0], rgb[:, 1]
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone.forward(rgb1)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.backbone.forward(rgb2)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        elif self.num_img == 1:
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone.forward(rgb)
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress.forward(feat, state)
            else:
                feat = feat.flatten(1, -1)
                feat = self.compress(feat)
        else:
            raise NotImplementedError(f"num_img={self.num_img} currently supports only 1 or 2")
        return torch.cat([feat, state], dim=-1)

    def forward(
        self,
        action,
        time,
        r,
        cond: dict,
        output_embedding=False,
        **kwargs,
    ):
        B, Ta, Da = action.shape
        action_flat = action.view(B, -1)
        cond_encoded = self._encode_cond(cond)

        if isinstance(time, (int, float)):
            time = torch.ones((B, 1), device=action.device) * time
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)
        r_emb = self._embed_r(r, B, action.device)

        features = torch.cat([action_flat, time_emb, cond_encoded], dim=-1)
        vel = self._apply_decoupled_head(features, r_emb)

        if output_embedding:
            return vel.view(B, Ta, Da), time_emb, r_emb, cond_encoded
        return vel.view(B, Ta, Da)

    def forward_encoder(self, cond: dict):
        return self._encode_cond(cond)

    @torch.no_grad()
    def sample_action(
        self,
        cond: dict,
        inference_steps: int,
        clip_intermediate_actions: bool,
        act_range: List[float],
        z: Tensor = None,
        save_chains: bool = False,
    ):
        B = cond["state"].shape[0]
        device = cond["state"].device
        x_hat = z if z is not None else torch.randn(B, self.horizon_steps, self.action_dim, device=device)
        if save_chains:
            x_chain = torch.zeros((B, inference_steps + 1, self.horizon_steps, self.action_dim), device=device)
            x_chain[:, 0] = x_hat

        t_vals = torch.linspace(0.0, 1.0, inference_steps + 1, device=device)
        for i in range(inference_steps):
            t_curr, r_next = t_vals[i], t_vals[i + 1]
            t = torch.full((B,), t_curr, device=device)
            r = torch.full((B,), r_next, device=device)
            u = self.forward(x_hat, t, r, cond)
            x_hat = x_hat + (r_next - t_curr) * u
            if clip_intermediate_actions or i == inference_steps - 1:
                x_hat = x_hat.clamp(*act_range)
            if save_chains:
                x_chain[:, i + 1] = x_hat

        if save_chains:
            return x_hat, x_chain
        return x_hat
