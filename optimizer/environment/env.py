import argparse
from typing import Tuple

import torch

from optimizer.environment.abstractenv import AbstractEnv
from optimizer.environment.spark.sparkcommunicator import SparkCommunicator


class Env(AbstractEnv):

    # Return state, reward, done
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        state = self.get_state()
        reward = self.communicator.act(action)
        done = self.communicator.is_done()
        return state, reward, done

    def _communicator(self, args: argparse.Namespace):
        return SparkCommunicator(args.rm_host, args.spark_history_server_host,
                                 args.hadoop_home, args.spark_home, args.java_home)
