import abc
import argparse
import logging

from optimizer.agent import Agent
from optimizer.replaymemory import ReplayMemory, MemorySerializer
from optimizer.util import fileutil


class AbstractController(object):

    STEP_SAVE_FILENAME = './results/steps.txt'

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = logging.getLogger(__name__)

        self.env = self._env(args)
        self.action_space = self.env.action_space()
        self.mem = ReplayMemory(self.args, self.args.memory_capacity)
        self.logger.info('Setup agent...')
        self.agent = Agent(self.args, self.env)
        self.logger.info('Agent setup finished.')
        self.priority_weight_increase = (1 - self.args.priority_weight) / (self.args.T_max - self.args.learn_start)
        self.t = 0

    @abc.abstractmethod
    def _env(self, args: argparse.Namespace):
        """Get Environment instance."""
        pass

    @abc.abstractmethod
    def run(self):
        pass

    def optimize_episode(self, state, act_func):
        self._reset_noise()

        action = act_func(state)
        state, reward, done = self.env.step(action)
        reward = self._clip_reward(reward)
        self.mem.append(state, action, reward, done)  # Append transition to memory
        self.log_step(action, reward)

        # Train
        if self.t >= self.args.learn_start:
            self._agent_learn()

        self.t += 1

        return state, action, reward, done

    def _load_memory(self) -> bool:
        mem_serializer = MemorySerializer(self.mem)
        return mem_serializer.try_load()

    def _clip_reward(self, reward):
        reward_clip = self.args.reward_clip
        if reward_clip > 0:
            reward = max(min(reward, reward_clip), -reward_clip)  # Clip rewards
        return reward

    def _reset_noise(self):
        if self.t % self.args.replay_frequency == 0:
            self.agent.reset_noise()  # Draw a new set of noisy weights

    def _agent_learn(self):
        mem, agent, env = self.mem, self.agent, self.env
        replay_frequency = self.args.replay_frequency
        target_update = self.args.target_update

        # Anneal importance sampling weight β to 1
        mem.priority_weight = min(mem.priority_weight + self.priority_weight_increase, 1)

        if self.t % replay_frequency == 0:
            agent.learn(mem, self.t)  # Train with n-step distributional double-Q learning
            self.logger.info('Agent learnt.')

        # Update target network
        if self.t % target_update == 0:
            agent.update_target_net()
            self.logger.info('Target net updated.')

    def log_step(self, action: int, reward: float):
        data = '%d,%d,%f' % (self.t, action, reward)
        fileutil.log_into_file(data, self.STEP_SAVE_FILENAME)
