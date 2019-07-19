import abc
import argparse
import logging

from optimizer.agent import Agent
from optimizer.controller.pretrainer import PreTrainer
from optimizer.replaymemory import ReplayMemoryProxy


class AbstractController(object):

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = logging.getLogger(__name__)

        self.env = self._env(args)
        self.action_space = self.env.action_space()
        self.mem = ReplayMemoryProxy(self.args, self.args.memory_capacity)
        self.agent = Agent(self.args, self.env)
        self.priority_weight_increase = (1 - self.args.priority_weight) / (self.args.T_max - self.args.learn_start)

    # Pre-train DQN model with offline data
    def pre_train_model(self):
        pre_trainer = PreTrainer(self.args, self.mem, self.agent,
                                 self.action_space, self.priority_weight_increase)
        pre_trainer.start_from_breakpoint()

    @abc.abstractmethod
    def _env(self, args: argparse.Namespace):
        """Get Environment instance."""
        pass

    @abc.abstractmethod
    def run(self):
        pass
