# MIT License

"""
Decoupled MeanFlow wrapper for ReFlow-style policies.

This keeps ReinFlow's noise-to-data convention:
    x_t = t * x_data + (1 - t) * noise,  t in [0, 1].
Sampling therefore integrates from t=0 to t=1 with
    x_r = x_t + (r - t) * u_theta(x_t, t, r, cond).
"""

import logging
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.flow.mlp_dmf import DecoupledFlowMLP

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


def stopgrad(x: Tensor) -> Tensor:
    return x.detach()


def _expand_time(t: Tensor, x: Tensor) -> Tensor:
    return t.view(t.shape[0], *([1] * (x.ndim - 1)))


def _adaptive_l2_loss(error: Tensor, gamma: float = 0.5, c: float = 1e-3) -> Tensor:
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)))
    weight = 1.0 / (delta_sq + c).pow(1.0 - gamma)
    return (stopgrad(weight) * delta_sq).mean()


def _cauchy_loss(error: Tensor, c: float = 1e-3) -> Tensor:
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)))
    return torch.log(delta_sq + c).mean()


class DecoupledMeanFlow(nn.Module):
    """
    DMF policy for accelerating ReFlow sampling with average velocities.

    The loss mirrors Decoupled MeanFlow: independent FM and MF samples are used,
    with a boundary flow-matching term u(x_t,t,t) and a flow-map term trained by
    a JVP target u_tgt = v + (r - t) du/dt.
    """

    def __init__(
        self,
        network: DecoupledFlowMLP,
        device: torch.device,
        horizon_steps: int,
        action_dim: int,
        act_min: float,
        act_max: float,
        obs_dim: int,
        max_denoising_steps: int,
        seed: int,
        sample_t_type: str = "logitnormal",
        P_mean: float = 0.0,
        P_std: float = 1.0,
        P_mean_t: float = 0.4,
        P_std_t: float = 1.0,
        P_mean_r: float = -1.2,
        P_std_r: float = 1.0,
        flow_loss_weight: float = 0.5,
        meanflow_loss_weight: float = 0.5,
        flow_ratio: float = 0.0,
        loss_type: str = "mse",
        gamma: float = 0.5,
        c: float = 1e-3,
        clamp_training_actions: bool = False,
        sampling_shift: float = 1.0,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        if int(max_denoising_steps) <= 0:
            raise ValueError("max_denoising_steps must be a positive integer")
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.network = network.to(device)
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.data_shape = (self.horizon_steps, self.action_dim)
        self.act_range = (act_min, act_max)
        self.obs_dim = obs_dim
        self.max_denoising_steps = int(max_denoising_steps)

        self.sample_t_type = sample_t_type
        self.P_mean = P_mean
        self.P_std = P_std
        self.P_mean_t = P_mean_t
        self.P_std_t = P_std_t
        self.P_mean_r = P_mean_r
        self.P_std_r = P_std_r
        self.flow_loss_weight = flow_loss_weight
        self.meanflow_loss_weight = meanflow_loss_weight
        self.flow_ratio = flow_ratio
        self.loss_type = loss_type
        self.gamma = gamma
        self.c = c
        self.clamp_training_actions = clamp_training_actions
        self.sampling_shift = sampling_shift
        self.last_loss_info = {}

        if freeze_encoder:
            if not hasattr(self.network, "set_encoder_requires_grad"):
                raise ValueError("freeze_encoder=True requires a DMF network with set_encoder_requires_grad")
            self.network.set_encoder_requires_grad(False)

    def sample_time(
        self,
        batch_size: int,
        time_sample_type: str = None,
        mean: float = 0.0,
        std: float = 1.0,
        **kwargs,
    ) -> Tensor:
        time_sample_type = time_sample_type or self.sample_t_type
        if time_sample_type == "uniform":
            return torch.rand(batch_size, device=self.device)
        if time_sample_type in {"logitnormal", "lognormal"}:
            normal = torch.randn(batch_size, device=self.device) * std + mean
            return torch.sigmoid(normal)
        if time_sample_type == "beta":
            alpha = kwargs.get("alpha", 1.5)
            beta = kwargs.get("beta", 1.0)
            beta_sample = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(self.device)
            return kwargs.get("s", 0.999) * beta_sample
        raise ValueError(f"Unknown time_sample_type={time_sample_type}")

    def sample_time_pair(self, batch_size: int) -> tuple[Tensor, Tensor]:
        t1 = self.sample_time(batch_size, mean=self.P_mean_t, std=self.P_std_t)
        t2 = self.sample_time(batch_size, mean=self.P_mean_r, std=self.P_std_r)
        t = torch.minimum(t1, t2)
        r = torch.maximum(t1, t2)

        if self.flow_ratio > 0:
            n_boundary = int(self.flow_ratio * batch_size)
            if n_boundary > 0:
                idx = torch.randperm(batch_size, device=self.device)[:n_boundary]
                r[idx] = t[idx]
        return t, r

    def generate_trajectory(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        t_full = _expand_time(t, x1)
        return t_full * x1 + (1.0 - t_full) * x0

    def _velocity_target(self, x1: Tensor, x0: Tensor) -> Tensor:
        return x1 - x0

    def _loss_from_error(self, error: Tensor) -> Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(error, torch.zeros_like(error))
        if self.loss_type == "adaptive_l2":
            return _adaptive_l2_loss(error, gamma=self.gamma, c=self.c)
        if self.loss_type == "cauchy":
            return _cauchy_loss(error, c=self.c)
        raise ValueError(f"Unknown loss_type={self.loss_type}")

    def loss(self, x1: Tensor, cond: dict) -> Tensor:
        if self.clamp_training_actions:
            x1 = x1.clamp(*self.act_range)
        batch_size = x1.shape[0]

        # Boundary flow-matching loss, sampled independently from the MF pair.
        t_fm = self.sample_time(batch_size, mean=self.P_mean, std=self.P_std)
        x0_fm = torch.randn_like(x1)
        xt_fm = self.generate_trajectory(x1, x0_fm, t_fm)
        v_fm = self._velocity_target(x1, x0_fm)
        u_fm = self.network(xt_fm, t_fm, t_fm, cond)
        fm_loss = self._loss_from_error(u_fm - stopgrad(v_fm))

        # MeanFlow / flow-map loss.
        t_mf, r_mf = self.sample_time_pair(batch_size)
        x0_mf = torch.randn_like(x1)
        xt_mf = self.generate_trajectory(x1, x0_mf, t_mf)
        v_mf = self._velocity_target(x1, x0_mf)

        def network_fn(z, t_val, r_val):
            return self.network(z, t_val, r_val, cond)

        u, du_dt = torch.autograd.functional.jvp(
            network_fn,
            (xt_mf, t_mf, r_mf),
            (v_mf, torch.ones_like(t_mf), torch.zeros_like(r_mf)),
            create_graph=True,
        )
        u_target = v_mf + _expand_time(r_mf - t_mf, u) * du_dt
        mf_loss = self._loss_from_error(u - stopgrad(u_target))

        loss = self.flow_loss_weight * fm_loss + self.meanflow_loss_weight * mf_loss
        self.last_loss_info = {
            "fm_loss": float(fm_loss.detach().cpu()),
            "mf_loss": float(mf_loss.detach().cpu()),
            "loss": float(loss.detach().cpu()),
        }
        return loss

    def _time_grid(self, inference_steps: int, device: torch.device) -> Tensor:
        t = torch.linspace(0.0, 1.0, inference_steps + 1, device=device)
        if self.sampling_shift == 1.0:
            return t
        shift = self.sampling_shift
        return shift * t / (1.0 + (shift - 1.0) * t)

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        inference_steps: int,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True,
        z: torch.Tensor = None,
    ) -> Sample:
        B = cond["state"].shape[0]
        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)

        x_hat_list = None
        if record_intermediate:
            x_hat_list = torch.zeros((B, inference_steps + 1) + self.data_shape, device=self.device)
            x_hat_list[:, 0] = x_hat

        t_vals = self._time_grid(inference_steps, x_hat.device)
        for i in range(inference_steps):
            t_curr, r_next = t_vals[i], t_vals[i + 1]
            t = torch.full((B,), t_curr, device=x_hat.device)
            r = torch.full((B,), r_next, device=x_hat.device)
            u = self.network(x_hat, t, r, cond)
            x_hat = x_hat + (r_next - t_curr) * u
            if clip_intermediate_actions or i == inference_steps - 1:
                x_hat = x_hat.clamp(*self.act_range)
            if record_intermediate:
                x_hat_list[:, i + 1] = x_hat

        x_hat = x_hat.clamp(*self.act_range)
        return Sample(trajectories=x_hat, chains=x_hat_list)
