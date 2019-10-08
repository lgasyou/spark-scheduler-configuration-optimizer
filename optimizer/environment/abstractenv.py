import abc
import argparse
import logging
import time
from collections import deque
from typing import Tuple

import torch

from optimizer.environment.stateinvalidexception import StateInvalidException
from optimizer.hyperparameters import STATE_SHAPE


class AbstractEnv(object):

    def __init__(self, args: argparse.Namespace):
        self.logger = logging.getLogger(__name__)
        self.simulating = args.use_simulation_env
        self.device = args.device
        self.buffer_history_length = args.history_length
        self.state_buffer = deque([], maxlen=args.history_length)
        self.communicator = self._communicator(args)
        self.actions = self.communicator.action_set

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        reward = self.communicator.act(action)
        state = self.try_get_state()
        done = self.communicator.is_done()
        return state, reward, done

    def get_state(self) -> torch.Tensor:
        state = self.communicator.get_state_tensor().to(self.device)
        self.state_buffer.append(state)
        return torch.stack(list(self.state_buffer), 0)

    def try_get_state(self, retry_interval: int = 2) -> torch.Tensor:
        state = None
        while state is None:
            try:
                state = self.get_state()
            except StateInvalidException:
                self.logger.warning("State obtaining failed. Try again in %ds." % retry_interval)
                time.sleep(retry_interval)
        return state

    def action_space(self) -> int:
        return len(self.actions)

    def reset_buffer(self):
        for _ in range(self.buffer_history_length):
            self.state_buffer.append(torch.zeros(*STATE_SHAPE, device=self.device))

    def close(self):
        pass

    @abc.abstractmethod
    def _communicator(self, args: argparse.Namespace):
        """Get Communicator instance."""
        pass
