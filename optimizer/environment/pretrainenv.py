import argparse
import pathlib

import torch

from optimizer.environment.abstractenv import AbstractEnv
from optimizer.environment.yarn.yarnslscommunicator import YarnSlsCommunicator


class PreTrainEnv(AbstractEnv):
    """
    Used while pre-training.
    Uses Google traces as its input.
    """

    TRAIN_SET = 'data/trainingset'

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.t = 0

    def start_sls(self, file_index: int, action_index: int):
        filename = "{}/sls-jobs{}.json".format(self.TRAIN_SET, file_index)
        if not pathlib.Path(filename).exists():
            print("File:", filename, "doesn't exist.")

        self.communicator.set_dataset(filename)
        self.communicator.override_config(action_index)
        self._reset()

    def step(self):
        state = self.communicator.get_state_tensor().to(self.device)
        reward = self.communicator.get_reward()
        done = self.communicator.is_done()
        self.state_buffer.append(state)
        return torch.stack(list(self.state_buffer), 0), reward, done

    def save_tensor(self, t: torch.Tensor):
        import numpy as np
        np.savetxt('./results/state%d.csv' % self.t, t.numpy(), delimiter=',', fmt='%.2f')
        self.t += 1

    def _communicator(self, args: argparse.Namespace):
        return YarnSlsCommunicator(args.resource_manager_host, args.spark_history_server_host, args.hadoop_home)

    def _reset(self):
        self.reset_buffer()
        self.communicator.reset()
