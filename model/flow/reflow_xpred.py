# MIT License

"""
Action-prediction Rectified Flow policy.

This variant keeps the standard rectified-flow interpolation
    x_t = t * x_1 + (1 - t) * x_0
but interprets the network output as a direct prediction of the clean action
chunk x_1. The training loss is still measured in velocity-field space by
converting x_1 prediction to the implied rectified-flow velocity.
"""

from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import Tensor

from model.flow.reflow import ReFlow


Sample = namedtuple("Sample", "trajectories chains")


class XPredReFlow(ReFlow):
    """Rectified Flow with direct action-chunk prediction."""

    def __init__(
        self,
        *args,
        eps: float = 1e-4,
        t_eps: float = None,
        loss_fp32: bool = True,
        repeated_flow_samples: int = 1,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        clip_sample: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eps = float(eps if t_eps is None else t_eps)
        self.loss_fp32 = loss_fp32
        self.repeated_flow_samples = int(repeated_flow_samples)
        self.noise_beta_alpha = float(noise_beta_alpha)
        self.noise_beta_beta = float(noise_beta_beta)
        self.noise_s = float(noise_s)
        self.clip_sample = clip_sample

        if self.repeated_flow_samples <= 0:
            raise ValueError("repeated_flow_samples must be positive")

    def sample_time(self, batch_size: int, time_sample_type: str = "uniform", **kwargs) -> Tensor:
        if time_sample_type == "beta":
            alpha = kwargs.get("alpha", self.noise_beta_alpha)
            beta = kwargs.get("beta", self.noise_beta_beta)
            s = kwargs.get("s", self.noise_s)
            dist = torch.distributions.Beta(alpha, beta)
            sample = dist.sample((batch_size,)).to(self.device)
            return ((s - sample) / s).clamp(0.0, 1.0 - self.eps)
        return super().sample_time(batch_size, time_sample_type, **kwargs)

    def generate_target(self, x1: Tensor) -> tuple:
        if self.repeated_flow_samples > 1:
            x1 = x1.repeat(self.repeated_flow_samples, 1, 1)
        return super().generate_target(x1)

    def _match_cond_batch(self, obs: dict, batch_size: int) -> dict:
        state = obs.get("state", None)
        if state is None or state.shape[0] == batch_size:
            return obs

        if batch_size % state.shape[0] != 0:
            raise ValueError(
                f"Cannot repeat conditioning batch {state.shape[0]} to match target batch {batch_size}"
            )

        repeats = batch_size // state.shape[0]
        matched = {}
        for key, value in obs.items():
            if torch.is_tensor(value) and value.shape[0] == state.shape[0]:
                matched[key] = value.repeat((repeats,) + (1,) * (value.ndim - 1))
            else:
                matched[key] = value
        return matched

    def _as_fp32_cond(self, obs: dict) -> dict:
        return {
            key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in obs.items()
        }

    def loss(self, xt: torch.Tensor, t: torch.Tensor, obs: dict, v: torch.Tensor) -> torch.Tensor:
        """Predict x_1 directly, then optimize the implied velocity.

        Given x_t = t*x_1 + (1-t)*x_0 and v = x_1 - x_0:
            x_1 = x_t + (1-t) * v
            v_hat = (x_1_hat - x_t) / (1-t)
        """
        obs = self._match_cond_batch(obs, xt.shape[0])

        if self.loss_fp32:
            xt = xt.float()
            t = t.float()
            v = v.float()
            obs = self._as_fp32_cond(obs)
            device_type = xt.device.type if xt.device.type in ("cuda", "cpu") else "cuda"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                return self._loss_impl(xt, t, obs, v)
        return self._loss_impl(xt, t, obs, v)

    def _loss_impl(self, xt: torch.Tensor, t: torch.Tensor, obs: dict, v: torch.Tensor) -> torch.Tensor:
        x1_hat = self.network(xt, t, obs)
        if self.clip_sample:
            x1_hat = x1_hat.clamp(*self.act_range)
        one_minus_t = (1.0 - t).view(t.shape[0], 1, 1).clamp_min(self.eps)
        v_hat = (x1_hat - xt) / one_minus_t
        return F.mse_loss(input=v_hat, target=v)

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        inference_steps: int,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True,
        z: torch.Tensor = None,
    ) -> Sample:
        """Sample with Euler updates induced by direct x_1 predictions."""
        batch_size = cond["state"].shape[0]
        if record_intermediate:
            x_hat_list = torch.zeros(
                (inference_steps,) + self.data_shape, device=self.device
            )

        x_hat = (
            z
            if z is not None
            else torch.randn((batch_size,) + self.data_shape, device=self.device)
        )

        t_vals = torch.linspace(0.0, 1.0, inference_steps + 1, device=self.device)
        for i in range(inference_steps):
            t_curr = t_vals[i]
            t_next = t_vals[i + 1]
            t = torch.full((batch_size,), t_curr, device=self.device)

            x1_hat = self.network(x_hat, t, cond)
            one_minus_t = (1.0 - t_curr).clamp_min(self.eps)
            v_hat = (x1_hat - x_hat) / one_minus_t
            x_hat = x_hat + (t_next - t_curr) * v_hat

            if clip_intermediate_actions or i == inference_steps - 1:
                x_hat = x_hat.clamp(*self.act_range)
            if record_intermediate:
                x_hat_list[i] = x_hat

        return Sample(
            trajectories=x_hat,
            chains=x_hat_list if record_intermediate else None,
        )
