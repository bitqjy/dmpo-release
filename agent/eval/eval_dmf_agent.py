# MIT License

"""
Evaluate Decoupled MeanFlow low-dimensional policies.
"""

import logging

from agent.eval.eval_meanflow_agent import EvalMeanFlowAgent
from model.flow.dmf import DecoupledMeanFlow

log = logging.getLogger(__name__)


class EvalDecoupledMeanFlowAgent(EvalMeanFlowAgent):
    def infer(self, cond: dict, num_denoising_steps: int):
        self.model: DecoupledMeanFlow
        return super().infer(cond, num_denoising_steps)
