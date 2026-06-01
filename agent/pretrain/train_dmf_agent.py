# MIT License

"""
Pre-training agent for Decoupled MeanFlow policies.
"""

import logging

import torch

from agent.pretrain.train_agent import PreTrainAgent
from model.flow.dmf import DecoupledMeanFlow

log = logging.getLogger(__name__)


class TrainDecoupledMeanFlowAgent(PreTrainAgent):
    """Training agent for DMF policies adapted from ReFlow."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: DecoupledMeanFlow
        self.ema_model: DecoupledMeanFlow

        self.verbose_train = False
        self.verbose_loss = False
        self.verbose_test = True

        if self.test_in_mujoco:
            self.test_log_all = True
            self.only_test = False
            self.test_denoising_steps = cfg.get("test_denoising_steps", 4)
            self.test_clip_intermediate_actions = True
            self.test_model_type = "ema"

    def get_loss(self, batch_data):
        actions, cond = batch_data
        try:
            return self.model.loss(actions, cond)
        except Exception as err:
            if not self.cfg.get("fallback_to_fm_loss", False):
                raise
            log.warning(f"DMF loss failed ({err}); falling back to boundary flow-matching loss.")
            batch_size = actions.shape[0]
            t = self.model.sample_time(batch_size)
            noise = torch.randn_like(actions)
            xt = self.model.generate_trajectory(actions, noise, t)
            target = self.model._velocity_target(actions, noise)
            pred = self.model.network(xt, t, t, cond)
            return torch.nn.functional.mse_loss(pred, target)

    def inference(self, cond: dict):
        model = self.ema_model if self.test_model_type == "ema" else self.model
        return model.sample(
            cond,
            inference_steps=self.test_denoising_steps,
            record_intermediate=False,
            clip_intermediate_actions=self.test_clip_intermediate_actions,
        )
