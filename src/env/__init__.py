import argparse
import pathlib
from collections import deque
from typing import Tuple, Generator

import torch

from .communicator import Communicator
from .exceptions import StateInvalidException
from .yarn import YarnSchedulerCommunicator, Action

__all__ = ['Env', 'GoogleTraceEnv']


class Env(object):
    """
    High level environment implementation.
    """

    def __init__(self, args: argparse.Namespace):
        self.device = args.device
        self.buffer_history_length = args.history_length
        self.state_buffer = deque([], maxlen=args.history_length)
        sls_jobs_json = args.test_set + '/sls-jobs.json'
        self.communicator: Communicator = YarnSchedulerCommunicator(args.rm_host, args.hadoop_home, sls_jobs_json)
        self.actions = self.communicator.get_action_set()
        self.training = True    # Consistent with model training mode

    def reset(self) -> torch.Tensor:
        self.__reset_buffer()
        self.communicator.reset()
        state = self.communicator.get_state_tensor().to(self.device)
        self.state_buffer.append(state)
        return torch.stack(list(self.state_buffer), 0)

    # Return state, reward, done
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        state = self.communicator.get_state_tensor().to(self.device)
        reward = self.communicator.act(action)
        done = self.communicator.is_done()
        self.state_buffer.append(state)
        return torch.stack(list(self.state_buffer), 0), reward, done

    def get_total_time_cost(self) -> int:
        return self.communicator.get_total_time_cost()

    # Uses loss of life as terminal signal
    def train(self) -> None:
        self.training = True

    # Uses standard terminal signal
    def eval(self) -> None:
        self.training = False

    def action_space(self) -> int:
        return len(self.actions)

    def close(self) -> None:
        self.communicator.close()

    def __reset_buffer(self):
        for _ in range(self.buffer_history_length):
            self.state_buffer.append(torch.zeros(42, 42, device=self.device))


class GoogleTraceEnv(object):
    """
    Used while pre-train running.
    Uses Google traces as its input.
    """

    def __init__(self, args: argparse.Namespace):
        self.device = args.device
        self.buffer_history_length = args.history_length
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training_set_path = args.training_set
        self.communicator = YarnSchedulerCommunicator(args.rm_host, args.hadoop_home, sls_jobs_json='')

    def get_generator(self, index_end: int=13, hour_end: int=23) -> Generator:
        action_set = self.communicator.get_action_set()
        for action_index in action_set.keys():
            for i in range(1, index_end):
                for j in range(hour_end):
                    filename = "{}/{}/sls_jobs{}.json".format(self.training_set_path, i, j)
                    if not pathlib.Path(filename).exists():
                        continue

                    yield self.step(filename, action_index)
            break

    def step(self, filename, action_index: int) -> Tuple[torch.Tensor, Action, float, bool]:
        self.communicator.sls_jobs_json = filename
        self.communicator.override_scheduler_xml_with(action_index)
        self.__reset()
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

    def __reset_buffer(self):
        for _ in range(self.buffer_history_length):
            self.state_buffer.append(torch.zeros(42, 42, device=self.device))

    def __reset(self):
        self.__reset_buffer()
        self.communicator.reset()
