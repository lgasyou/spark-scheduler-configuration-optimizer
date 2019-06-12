import abc
import argparse
import logging

from ..agent import Agent
from ..pretrainer import PreTrainer
from ..replaymemory import ReplayMemory


class AbstractController(object):
    """
    Cluster Scheduler's Configuration Optimizer
    -----------------------------------------
    算法流程：
    1.	用随机权重初始化价值函数；
    2.	预训练DQN模型；
    3.	f设置为0；
    4.	for 片段(episode) 1 to
    5.	    for 时间片 1 to
    6.	       	在状态下用贪婪算法选一个动作；
    7.	 		执行动作并让调度器去观察奖励和下一个状态
    8.	 		把样本加到中；
    9.	 		if (f % F == 0)
    10.	 			使用中的样本训练深度学习模型DQN；
    11.	 		end if
    12.         f = f + 1
    13.	 	end for
    14.	end for

    When raises the exception "StateInvalidException",
    the state is invalid so state, action, reward of this step, signal terminate
    won't be saved into the memory.
    This will cause a problem: non-terminate signal will always be True.
    TODO: See description above.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = logging.getLogger(__name__)

        self.env = self._env(args)
        self.action_space = self.env.action_space()
        self.mem = ReplayMemory(self.args, self.args.memory_capacity)
        self.agent = Agent(self.args, self.env)
        self.priority_weight_increase = (1 - self.args.priority_weight) / (self.args.T_max - self.args.learn_start)

    # Pre-train DQN model with offline data
    def pre_train_model(self):
        pre_trainer = PreTrainer(self.mem, self.agent, self.args)
        pre_trainer.start_pre_train()

    @abc.abstractmethod
    def _env(self, args: argparse.Namespace):
        """Get Environment instance."""
        pass

    @abc.abstractmethod
    def run(self):
        pass
