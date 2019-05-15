import argparse
import pathlib
from collections import deque
from typing import Tuple, Generator

import torch

from .exceptions import StateInvalidException
from .yarncommunicator import YarnCommunicator, YarnSlsCommunicator
from .yarnmodel import Action
from ..hyperparameters import STATE_SHAPE

__all__ = ['ReadOnlyEnv', 'PreTrainEnv']


class ReadOnlyEnv(object):

    def __init__(self, args: argparse.Namespace, env_communicator):
        self.device = args.device
        self.buffer_history_length = args.history_length
        self.state_buffer = deque([], maxlen=args.history_length)
        self.communicator = env_communicator
        self.actions = self.communicator.get_action_set()

    def get_state(self) -> torch.Tensor:
        state = self.communicator.get_state_tensor().to(self.device)
        self.state_buffer.append(state)
        return torch.stack(list(self.state_buffer), 0)

    def action_space(self) -> int:
        return len(self.actions)


class PreTrainEnv(ReadOnlyEnv):
    """
    Used while pre-train running.
    Uses Google traces as its input.
    """

    def __init__(self, args: argparse.Namespace):
        env_communicator = YarnSlsCommunicator(args.rm_host, args.hadoop_home, sls_jobs_json='')
        super().__init__(args, env_communicator)
        self.training_set_path = args.training_set

    def get_generator(self, index_end: int = 13, hour_end: int = 23) -> Generator:
        action_set = self.communicator.get_action_set()
        for action_index in action_set.keys():
            for i in range(1, index_end):
                for j in range(hour_end):
                    filename = "{}/{}/sls_jobs{}.json".format(self.training_set_path, i, j)
                    if not pathlib.Path(filename).exists():
                        continue

                    yield self.step(filename, action_index)
            break   # For fasten test. Should be removed finally.

    def step(self, filename, action_index: int) -> Tuple[torch.Tensor, Action, float, bool]:
        self.communicator.sls_jobs_json = filename
        self.communicator.override_scheduler_xml_with(action_index)
        self._reset()
        done = False

        while not done:
            try:
                state = self.communicator.get_state_tensor().to(self.device)
            except StateInvalidException:
                return

            reward = self.communicator.get_reward()
            done = self.communicator.is_done()
            self.state_buffer.append(state)
            yield torch.stack(list(self.state_buffer), 0), action_index, reward, done

    def _reset_buffer(self):
        for _ in range(self.buffer_history_length):
            self.state_buffer.append(torch.zeros(*STATE_SHAPE, device=self.device))

    def _reset(self):
        self._reset_buffer()
        self.communicator.reset()
