import abc
import argparse
import logging
import time

from .agent import Agent
from .environment import Env
from .environment import PreTrainEnv
from .environment.exceptions import StateInvalidException
from .memory import ReplayMemory


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
        self.dqn = Agent(self.args, self.env)
        self.priority_weight_increase = (1 - self.args.priority_weight) / (self.args.T_max - self.args.learn_start)

    # Pre-train DQN model with offline data
    def pre_train_model(self):
        self.logger.info('Pre-training DQN model...')

        # Try to load data from file, if fails run training set and
        # save them into memory.
        if not self.mem.try_load_from_file():
            train_env = PreTrainEnv(self.args)
            # Get data generator
            generator = train_env.get_generator()

            T = 0
            # Generate data, then save them into self.mem
            for step, action_index, hour in generator:
                for (state, action, reward, terminal) in step:
                    print('Iteration: %d, Reward: %f' % (T, reward))
                    self.mem.append(state, action, reward, terminal)
                    time.sleep(5)
                    T += 1

                self.mem.save('./results/pre-train-replay-memory-%d-%d.pk' % (action_index, hour))

            # Save data as 'pre-train-replay-memory.pk'
            self.mem.save()

        # Pre-train DQN model by using training set
        self.dqn.learn(self.mem)
        self.logger.info('Pre-training DQN model finished.')

    @abc.abstractmethod
    def _env(self, args: argparse.Namespace):
        """
        Get Environment instance.
        """
        pass

    @abc.abstractmethod
    def run(self):
        pass


class OptimizationController(AbstractController):

    def run(self):
        env = self.env
        args = self.args
        dqn = self.dqn
        mem = self.mem

        priority_weight_increase = self.priority_weight_increase
        reward_clip = args.reward_clip

        T = 0
        state = env.get_state()
        while True:
            action = dqn.act(state)

            try:
                next_state, reward, done = env.step(action)  # Step
                if reward_clip > 0:
                    reward = max(min(reward, reward_clip), -reward_clip)  # Clip rewards
                mem.append(state, action, reward, done)  # Append transition to memory
                time.sleep(1)
                T += 1
            except StateInvalidException:
                break

            if done:
                break

            if T % args.replay_frequency == 0:
                dqn.reset_noise()  # Draw a new set of noisy weights

            # Train and test
            if T >= args.learn_start:
                # Anneal importance sampling weight β to 1
                mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

                if T % args.replay_frequency == 0:
                    dqn.learn(mem)  # Train with n-step distributional double-Q learning

                # Update target network
                if T % args.target_update == 0:
                    dqn.update_target_net()
            state = next_state

    def _env(self, args: argparse.Namespace):
        return Env(args)
