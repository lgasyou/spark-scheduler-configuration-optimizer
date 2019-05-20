import abc
import argparse
import pathlib
from collections import deque
from typing import Tuple, Generator

import torch

from .exceptions import StateInvalidException
from .yarncommunicator import YarnCommunicator, YarnSlsCommunicator
from .yarnmodel import Action
from ..hyperparameters import STATE_SHAPE

__all__ = ['AbstractEnv', 'Env', 'PreTrainEnv']


class AbstractEnv(object):

    def __init__(self, args: argparse.Namespace):
        self.device = args.device
        self.buffer_history_length = args.history_length
        self.state_buffer = deque([], maxlen=args.history_length)
        self.communicator = self._communicator(args)
        self.actions = self.communicator.get_action_set()

    def get_state(self) -> torch.Tensor:
        state = self.communicator.get_state_tensor().to(self.device)
        self.state_buffer.append(state)
        return torch.stack(list(self.state_buffer), 0)

    def action_space(self) -> int:
        return len(self.actions)

    @abc.abstractmethod
    def _communicator(self, args: argparse.Namespace):
        """
        Get Communicator instance.
        """
        pass


class Env(AbstractEnv):

    def _communicator(self, args: argparse.Namespace):
        return YarnCommunicator(args.rm_host, args.hadoop_home)


class PreTrainEnv(AbstractEnv):
    """
    Used while pre-training.
    Uses Google traces as its input.
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.training_set_path = args.training_set

    def get_generator(self, index_end: int = 24) -> Generator:
        for action_index in [2]:
            for i in range(1, index_end):
                filename = "{}/sls-jobs{}.json".format(self.training_set_path, i)
                if not pathlib.Path(filename).exists():
                    continue

                yield self.step(filename, action_index), action_index, i

    def step(self, filename, action_index: int) -> Tuple[torch.Tensor, Action, float, bool]:
        self.communicator.set_sls_jobs_json(filename)
        self.communicator.override_configuration(action_index)
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

    def _communicator(self, args: argparse.Namespace):
        return YarnSlsCommunicator(args.rm_host, args.hadoop_home)

    def _reset_buffer(self):
        for _ in range(self.buffer_history_length):
            self.state_buffer.append(torch.zeros(*STATE_SHAPE, device=self.device))

    def _reset(self):
        self._reset_buffer()
        self.communicator.reset()
