import argparse
import random
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.agent import IAgent, RainbowAgent
from src.env import IEnv, RainbowEnv
from src.memory import ReplayMemory
from src.test import test


def read_data_from_csv(path, index, hour) -> np.ndarray:
    return np.loadtxt("{}/{}/sls_jobs{}.csv".format(path, index, hour), delimiter=",", dtype=np.uint8)


# Simple ISO 8601 timestamped logger
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


class ClusterSchedConfOptimizer(object):
    """
    Cluster Scheduler Configuration Optimizer
    """

    def __init__(self, data_path: str, args: argparse.Namespace):
        self.data_path = data_path
        self.args = args
        self.env, self.action_space = self.__setup_env()
        self.dqn, self.mem, self.priority_weight_increase = self.__setup_agent()
        self.val_mem = self.__setup_val_mem()
        self.__read_training_jobs()

    # Environment
    def __setup_env(self) -> Tuple[IEnv, int]:
        env = RainbowEnv(self.args)
        env.train()
        action_space = env.action_space()
        return env, action_space

    # Agent
    def __setup_agent(self) -> Tuple[IAgent, ReplayMemory, float]:
        dqn = RainbowAgent(self.args, self.env)
        mem = ReplayMemory(self.args, self.args.memory_capacity)
        priority_weight_increase = (1 - self.args.priority_weight) / (self.args.T_max - self.args.learn_start)
        return dqn, mem, priority_weight_increase

    # Construct validation memory
    def __setup_val_mem(self) -> ReplayMemory:
        val_mem = ReplayMemory(self.args, self.args.evaluation_size)
        T, done = 0, True
        while T < self.args.evaluation_size:
            if done:
                state, done = self.env.reset(), False

            next_state, _, done = self.env.step(random.randint(0, self.action_space - 1))
            val_mem.append(state, None, None, done)
            state = next_state
            T += 1
        return val_mem

    def __read_training_jobs(self) -> None:
        for i in range(1, 13):
            for j in range(0, 23):
                np_data = read_data_from_csv(self.data_path, i, j)
                tensor = torch.from_numpy(np_data)
                self.mem.append(state=tensor, action=None, reward=None, terminal=False)

    # Training loop
    def start(self) -> None:
        num_training_steps = self.args.T_max
        reward_clip = self.args.reward_clip

        T, done = 0, True
        self.dqn.train()
        for T in tqdm(range(num_training_steps)):
            if done:
                state, done = self.env.reset(), False

            if T % self.args.replay_frequency == 0:
                self.dqn.reset_noise()  # Draw a new set of noisy weights

            action = self.dqn.act(state)  # Choose an action greedily (with noisy weights)
            next_state, reward, done = self.env.step(action)  # Step
            if reward_clip > 0:
                reward = max(min(reward, reward_clip), -reward_clip)  # Clip rewards
            self.mem.append(state, action, reward, done)  # Append transition to memory
            T += 1

            # Train and test
            if T >= self.args.learn_start:
                # Anneal importance sampling weight β to 1
                self.mem.priority_weight = min(self.mem.priority_weight + self.priority_weight_increase, 1)

                if T % self.args.replay_frequency == 0:
                    self.dqn.learn(self.mem)  # Train with n-step distributional double-Q learning

                if T % self.args.evaluation_interval == 0:
                    self.dqn.eval()  # Set DQN (online network) to evaluation mode
                    avg_reward, avg_Q = test(self.args, T, self.dqn, self.val_mem)  # Test
                    log('T = ' + str(T) + ' / ' + str(num_training_steps) + ' | Avg. reward: ' +
                        str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                    self.dqn.train()  # Set DQN (online network) back to training mode

                # Update target network
                if T % self.args.target_update == 0:
                    self.dqn.update_target_net()
            state = next_state

        self.env.close()
