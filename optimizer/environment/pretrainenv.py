import argparse

import torch

from optimizer.environment.abstractenv import AbstractEnv
from optimizer.environment.spark.sparktrainingcommunicator import SparkTrainingCommunicator


class PreTrainEnv(AbstractEnv):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def generate_pre_train_set(self) -> dict:
        return self.communicator.generate_pre_train_set()

    def start(self, action_index: int, workloads):
        self.communicator.override_config(action_index)
        self._reset(workloads)

    def step(self):
        state = self.communicator.get_state_tensor().to(self.device)
        reward = self.communicator.get_reward()
        done = self.communicator.is_done()
        self.state_buffer.append(state)
        return torch.stack(list(self.state_buffer), 0), reward, done

    def _communicator(self, args: argparse.Namespace):
        return SparkTrainingCommunicator(args.resource_manager_host, args.spark_history_server_host,
                                         args.hadoop_home, args.spark_home, args.java_home)

    def _reset(self, workloads):
        self.reset_buffer()
        self.communicator.reset(workloads)
