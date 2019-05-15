import argparse
from typing import Tuple

import torch

from ..environment import AbstractEnv
from ..environment.yarncommunicator import YarnSlsCommunicator
from ..hyperparameters import STATE_SHAPE


class EvaluationEnv(AbstractEnv):
    """
    High level environment implementation.
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.training = True  # Consistent with model training mode

    def reset(self) -> torch.Tensor:
        self._reset_buffer()
        self.communicator.reset()
        state = self.communicator.get_state_tensor().to(self.device)
        self.state_buffer.append(state)
        return torch.stack(list(self.state_buffer), 0)

    # Return state, reward, done
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        state = self.get_state()
        reward = self.communicator.act(action)
        done = self.communicator.is_done()
        return state, reward, done

    def get_total_time_cost(self) -> tuple:
        return self.communicator.get_total_time_cost()

    # Uses loss of life as terminal signal
    def train(self) -> None:
        self.training = True

    # Uses standard terminal signal
    def eval(self) -> None:
        self.training = False

    def close(self) -> None:
        self.communicator.close()

    def _communicator(self, args: argparse.Namespace):
        sls_jobs_json = args.test_set + '/sls-jobs.json'
        return YarnSlsCommunicator(args.rm_host, args.hadoop_home, sls_jobs_json)

    def _reset_buffer(self):
        for _ in range(self.buffer_history_length):
            self.state_buffer.append(torch.zeros(*STATE_SHAPE, device=self.device))
