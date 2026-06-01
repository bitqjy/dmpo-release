# MIT License

"""
Evaluate Decoupled MeanFlow image-conditioned policies.
"""

import logging

from agent.eval.eval_meanflow_img_agent import EvalImgMeanFlowAgent
from model.flow.dmf import DecoupledMeanFlow

log = logging.getLogger(__name__)


class EvalImgDecoupledMeanFlowAgent(EvalImgMeanFlowAgent):
    def infer(self, cond: dict, num_denoising_steps: int):
        self.model: DecoupledMeanFlow
        return super().infer(cond, num_denoising_steps)
