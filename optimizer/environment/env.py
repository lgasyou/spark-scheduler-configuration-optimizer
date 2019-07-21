import argparse
from typing import Tuple

import torch

from optimizer.environment.abstractenv import AbstractEnv
from optimizer.environment.clustercommunication.communicator import Communicator


class Env(AbstractEnv):

    # Return state, reward, done
    def step(self, action: int, retry_interval: int) -> Tuple[torch.Tensor, float, bool]:
        state = self.try_get_state(retry_interval)
        reward = self.communicator.act(action)
        done = self.communicator.is_done()
        return state, reward, done

    def _communicator(self, args: argparse.Namespace):
        return Communicator(args.resource_manager_host, args.spark_history_server_host, args.hadoop_home)
