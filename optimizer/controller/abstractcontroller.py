import abc
import argparse
import logging

from optimizer.agent import Agent
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
        self.t = 0

    @abc.abstractmethod
    def _env(self, args: argparse.Namespace):
        """Get Environment instance."""
        pass

    @abc.abstractmethod
    def run(self):
        pass

    def optimize_episode(self, state, act_func, interval):
        self._reset_noise()

        action = act_func(state)
        state, reward, done = self.env.step(action, interval)
        reward = self._clip_reward(reward)
        self.mem.append(state, action, reward, done)  # Append transition to memory

        if done:
            self.mem.terminate()
        self.t += 1

        # Train
        if self.t >= self.args.learn_start:
            self._agent_learn()

        return state, action, reward, done

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
            agent.learn(mem)  # Train with n-step distributional double-Q learning

        # Update target network
        if self.t % target_update == 0:
            agent.update_target_net()
