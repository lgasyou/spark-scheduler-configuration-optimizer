import abc
import argparse
from collections import deque

import torch


class AbstractEnv(object):

    def __init__(self, args: argparse.Namespace):
        self.device = args.device
        self.buffer_history_length = args.history_length
        self.state_buffer = deque([], maxlen=args.history_length)
        self.communicator = self._communicator(args)
        self.actions = self.communicator.action_set

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
