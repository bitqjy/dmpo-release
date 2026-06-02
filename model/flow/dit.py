# MIT License

"""
DiT-style action heads for ReFlow and Decoupled MeanFlow.

These modules are action-token transformers: the noisy action trajectory is
projected into horizon tokens, conditioned by observation features and timestep
embeddings through AdaLN-Zero blocks, and decoded back to per-step velocities.
"""

import logging
import math
from copy import deepcopy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from model.common.modules import RandomShiftsAug, SpatialEmb

if TYPE_CHECKING:
    from model.common.vit import VitEncoder

log = logging.getLogger(__name__)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def zero_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        nn.init.zeros_(param)
    return module


def set_requires_grad(module: nn.Module, flag: bool):
    for param in module.parameters():
        param.requires_grad = flag


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32) + self.eps)
        return x * norm.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.shape[0])
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.frequency_embedding = SinusoidalPosEmb(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.frequency_embedding(t))


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        qk_norm: bool = False,
        ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads, dropout=dropout, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        if activation == "gelu":
            act = nn.GELU(approximate="tanh")
        elif activation == "silu":
            act = nn.SiLU()
        elif activation == "mish":
            act = nn.Mish()
        else:
            raise ValueError(f"Unsupported activation={activation}")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            act,
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        zero_module(self.adaLN_modulation[-1])

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(attn_in)
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, action_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        zero_module(self.adaLN_modulation[-1])
        zero_module(self.linear)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class ActionDiT(nn.Module):
    """
    Low-dimensional DiT velocity head for ReFlow.

    Inputs:
        action: (B, horizon_steps, action_dim)
        time:   (B,)
        cond:   {"state": (B, cond_steps, obs_dim)}
    Output:
        velocity: (B, horizon_steps, action_dim)
    """

    def __init__(
        self,
        action_dim: int,
        horizon_steps: int,
        cond_dim: int,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        time_frequency_dim: int = 256,
        dmf_depth: int = None,
        qk_norm: bool = False,
        use_logvar: bool = False,
        logvar_frequency_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.cond_dim = cond_dim
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.dmf_depth = depth if dmf_depth is None else int(dmf_depth)
        self.use_logvar = use_logvar

        self.x_embedder = nn.Linear(action_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon_steps, hidden_size))
        self.time_embedder = TimestepEmbedder(hidden_size, time_frequency_dim)
        self.cond_embedder = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    activation=activation,
                    qk_norm=qk_norm,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, action_dim)
        if use_logvar:
            self.logvar_t_embedder = SinusoidalPosEmb(logvar_frequency_dim)
            self.logvar_linear = nn.Linear(logvar_frequency_dim * 2, 1)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)
        if self.use_logvar:
            nn.init.zeros_(self.logvar_linear.weight)
            nn.init.zeros_(self.logvar_linear.bias)

    def encode_condition(self, cond: dict) -> Tensor:
        state = cond["state"].view(cond["state"].shape[0], -1)
        return state

    def _condition_embedding(self, time: Tensor, cond: dict) -> Tensor:
        B = cond["state"].shape[0]
        if isinstance(time, (int, float)):
            time = torch.full((B,), float(time), device=cond["state"].device)
        time_emb = self.time_embedder(time.view(B))
        cond_emb = self.cond_embedder(self.encode_condition(cond))
        return time_emb + cond_emb

    def _forward_with_conditions(
        self,
        action: Tensor,
        c_t: Tensor,
        c_r: Tensor = None,
        dmf_depth: int = None,
    ) -> Tensor:
        x = self.x_embedder(action) + self.pos_embed
        if c_r is None:
            for block in self.blocks:
                x = block(x, c_t)
            return self.final_layer(x, c_t)

        split = self.dmf_depth if dmf_depth is None else int(dmf_depth)
        split = max(0, min(split, len(self.blocks)))
        for block in self.blocks[:split]:
            x = block(x, c_t)
        for block in self.blocks[split:]:
            x = block(x, c_r)
        return self.final_layer(x, c_r)

    def _logvar(self, time: Tensor, r: Tensor, action: Tensor) -> Tensor:
        if not self.use_logvar:
            raise ValueError("return_logvar=True requires use_logvar=True")
        if r is None:
            r = time
        emb = torch.cat(
            [
                self.logvar_t_embedder(time.view(time.shape[0])),
                self.logvar_t_embedder(r.view(r.shape[0])),
            ],
            dim=1,
        )
        logvar = self.logvar_linear(emb)
        return logvar.view(-1, *([1] * (action.ndim - 1)))

    def forward(
        self,
        action: Tensor,
        time: Tensor,
        cond: dict,
        return_logvar: bool = False,
        **kwargs,
    ) -> Tensor:
        c_t = self._condition_embedding(time, cond)
        out = self._forward_with_conditions(action, c_t)
        if return_logvar:
            return out, self._logvar(time, None, out)
        return out

    def set_encoder_requires_grad(self, flag: bool):
        set_requires_grad(self.x_embedder, flag)
        self.pos_embed.requires_grad = flag
        set_requires_grad(self.cond_embedder, flag)
        split = max(0, min(self.dmf_depth, len(self.blocks)))
        for block in self.blocks[:split]:
            set_requires_grad(block, flag)


class DecoupledActionDiT(ActionDiT):
    """DiT flow-map head: encoder blocks use t, decoder blocks use r."""

    def forward(
        self,
        action: Tensor,
        time: Tensor,
        r: Tensor,
        cond: dict,
        return_logvar: bool = False,
        **kwargs,
    ) -> Tensor:
        c_t = self._condition_embedding(time, cond)
        c_r = self._condition_embedding(r, cond)
        out = self._forward_with_conditions(action, c_t, c_r)
        if return_logvar:
            return out, self._logvar(time, r, out)
        return out


class VisionActionDiT(ActionDiT):
    """Image-conditioned DiT velocity head for ReFlow."""

    def __init__(
        self,
        backbone: "VitEncoder",
        action_dim: int,
        horizon_steps: int,
        cond_dim: int,
        img_cond_steps: int = 1,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        time_frequency_dim: int = 256,
        visual_feature_dim: int = 128,
        spatial_emb: int = 0,
        num_img: int = 1,
        augment: bool = False,
        dmf_depth: int = None,
        qk_norm: bool = False,
        use_logvar: bool = False,
        logvar_frequency_dim: int = 128,
    ):
        if spatial_emb > 0:
            assert spatial_emb > 1, "spatial_emb must be > 1"
            if num_img not in [1, 2]:
                raise NotImplementedError(f"num_img={num_img} currently supports only 1 or 2")
            cond_enc_dim = spatial_emb * num_img + cond_dim
        else:
            cond_enc_dim = visual_feature_dim * num_img + cond_dim

        super().__init__(
            action_dim=action_dim,
            horizon_steps=horizon_steps,
            cond_dim=cond_enc_dim,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            activation=activation,
            time_frequency_dim=time_frequency_dim,
            dmf_depth=dmf_depth,
            qk_norm=qk_norm,
            use_logvar=use_logvar,
            logvar_frequency_dim=logvar_frequency_dim,
        )

        self.backbone = backbone
        self.img_cond_steps = img_cond_steps
        self.num_img = num_img
        self.augment = augment
        self.prop_dim = cond_dim
        self.spatial_emb = spatial_emb
        if augment:
            self.aug = RandomShiftsAug(pad=4)

        if spatial_emb > 0:
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
        else:
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )

    def encode_condition(self, cond: dict) -> Tensor:
        import einops

        B = cond["state"].shape[0]
        state = cond["state"].view(B, -1)
        rgb = cond["rgb"][:, -self.img_cond_steps :]
        _, T_rgb, C, H, W = rgb.shape

        if self.num_img > 1:
            channels_per_img = C // self.num_img
            rgb = rgb.reshape(B, T_rgb, self.num_img, channels_per_img, H, W)
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
            feat2 = self.backbone.forward(rgb2)
            if self.spatial_emb > 0:
                feat1 = self.compress1.forward(feat1, state)
                feat2 = self.compress2.forward(feat2, state)
            else:
                feat1 = self.compress(feat1.flatten(1, -1))
                feat2 = self.compress(feat2.flatten(1, -1))
            feat = torch.cat([feat1, feat2], dim=-1)
        elif self.num_img == 1:
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone.forward(rgb)
            if self.spatial_emb > 0:
                feat = self.compress.forward(feat, state)
            else:
                feat = self.compress(feat.flatten(1, -1))
        else:
            raise NotImplementedError(f"num_img={self.num_img} currently supports only 1 or 2")

        return torch.cat([feat, state], dim=-1)

    def set_encoder_requires_grad(self, flag: bool):
        super().set_encoder_requires_grad(flag)
        set_requires_grad(self.backbone, flag)
        for name in ["compress", "compress1", "compress2"]:
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                set_requires_grad(module, flag)


class DecoupledVisionActionDiT(VisionActionDiT):
    """Image-conditioned DiT flow-map head with t/r block decoupling."""

    def forward(
        self,
        action: Tensor,
        time: Tensor,
        r: Tensor,
        cond: dict,
        return_logvar: bool = False,
        **kwargs,
    ) -> Tensor:
        c_t = self._condition_embedding(time, cond)
        c_r = self._condition_embedding(r, cond)
        out = self._forward_with_conditions(action, c_t, c_r)
        if return_logvar:
            return out, self._logvar(time, r, out)
        return out
