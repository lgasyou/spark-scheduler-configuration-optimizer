import argparse
from typing import Tuple

import torch

from optimizer.environment import AbstractEnv
from optimizer.environment.spark.sparkcommunicator import SparkCommunicator
from optimizer.hyperparameters import STATE_SHAPE


class EvaluationEnv(AbstractEnv):
    """
    High level environment implementation.
    """

    TEST_SET = 'data/testset'

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.training = True  # Consistent with model training mode

    def reset(self) -> torch.Tensor:
        self.reset_buffer()
        self.communicator.reset()
        return self.get_state()

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
        return SparkCommunicator(args.rm_host, args.spark_history_server_host,
                                 args.hadoop_home, args.spark_home, args.java_home)
