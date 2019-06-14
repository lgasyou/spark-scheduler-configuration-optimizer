import argparse
import pathlib

import torch

from .abstractenv import AbstractEnv
from .yarnenvironment.yarnslscommunicator import YarnSlsCommunicator
from ..hyperparameters import STATE_SHAPE


class PreTrainEnv(AbstractEnv):
    """
    Used while pre-training.
    Uses Google traces as its input.
    """

    TRAIN_SET = 'data/trainingset'

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def start_sls(self, file_index: int, action_index: int):
        filename = "{}/sls-jobs{}.json".format(self.TRAIN_SET, file_index)
        if not pathlib.Path(filename).exists():
            print("File:", filename, "doesn't exist.")

        self.communicator.set_dataset(filename)
        self.communicator.override_configuration(action_index)
        self._reset()

    def step(self):
        state = self.communicator.get_state_tensor().to(self.device)
        reward = self.communicator.get_reward()
        done = self.communicator.is_done()
        self.state_buffer.append(state)
        return torch.stack(list(self.state_buffer), 0), reward, done

    def _communicator(self, args: argparse.Namespace):
        return YarnSlsCommunicator(args.rm_host, args.hadoop_home)

    def _reset_buffer(self):
        for _ in range(self.buffer_history_length):
            self.state_buffer.append(torch.zeros(*STATE_SHAPE, device=self.device))

    def _reset(self):
        self._reset_buffer()
        self.communicator.reset()
