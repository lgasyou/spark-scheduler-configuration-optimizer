import argparse
import random
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from src.agent import Agent
from src.env import Env
from src.memory import ReplayMemory
from src.test import test


def read_data_from_csv(path, index, hour) -> np.ndarray:
    data = np.loadtxt("{}/{}/sls_jobs{}.csv".format(path, index, hour), delimiter=",", dtype=np.uint8)
    return data


# Simple ISO 8601 timestamped logger
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


# Cluster Scheduler Configuration Optimizer
class CSCOptimizer(object):

    def __init__(self, data_path: str, args: argparse.Namespace):
        self.data_path = data_path
        self.args = args
        self.env, self.action_space = self.setup_env()
        self.dqn, self.mem, self.priority_weight_increase = self.setup_agent()
        self.read_training_jobs()

    # Environment
    def setup_env(self) -> tuple:
        env = Env(self.args)
        env.train()
        action_space = env.action_space()
        return env, action_space

    # Agent
    def setup_agent(self) -> tuple:
        dqn = Agent(self.args, self.env)
        mem = ReplayMemory(self.args, self.args.memory_capacity)
        priority_weight_increase = (1 - self.args.priority_weight) / (self.args.T_max - self.args.learn_start)
        return dqn, mem, priority_weight_increase

    def read_training_jobs(self) -> None:
        for i in range(1, 13):
            for j in range(0, 23):
                np_data = read_data_from_csv(self.data_path, i, j)
                tensor = torch.from_numpy(np_data)
                self.mem.append(state=tensor, action=None, reward=None, terminal=False)

    def start(self):
        # Construct validation memory
        val_mem = ReplayMemory(self.args, self.args.evaluation_size)
        T, done = 0, True
        while T < self.args.evaluation_size:
            if done:
                state, done = self.env.reset(), False

            next_state, _, done = self.env.step(random.randint(0, self.action_space - 1))
            val_mem.append(state, None, None, done)
            state = next_state
            T += 1

        # Training loop
        T, done = 0, True
        self.dqn.train()
        num_training_steps = self.args.T_max
        # num_training_steps = 1  # debug
        for T in tqdm(range(num_training_steps)):
            if done:
                state, done = self.env.reset(), False

            if T % self.args.replay_frequency == 0:
                self.dqn.reset_noise()  # Draw a new set of noisy weights

            action = self.dqn.act(state)  # Choose an action greedily (with noisy weights)
            next_state, reward, done = self.env.step(action)  # Step
            if self.args.reward_clip > 0:
                reward = max(min(reward, self.args.reward_clip), -self.args.reward_clip)  # Clip rewards
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
                    avg_reward, avg_Q = test(self.args, T, self.dqn, val_mem)  # Test
                    log('T = ' + str(T) + ' / ' + str(self.args.T_max) + ' | Avg. reward: ' +
                        str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                    self.dqn.train()  # Set DQN (online network) back to training mode

                # Update target network
                if T % self.args.target_update == 0:
                    self.dqn.update_target_net()
            state = next_state

            # Anneal importance sampling weight β to 1
            # self.mem.priority_weight = min(self.mem.priority_weight + self.priority_weight_increase, 1)
            # if T % self.args.replay_frequency == 0:
            #     self.dqn.learn(self.mem)  # Train with n-step distributional double-Q learning
            #     self.dqn.eval()  # Set DQN (online network) to evaluation mode
            #     avg_reward, avg_Q = test(self.args, 0, self.dqn, val_mem, evaluate=True)  # Test
            #     self.dqn.train()
            #
            # # Update target network
            # if T % self.args.target_update == 0:
            #     self.dqn.update_target_net()
            # state = next_state

        self.env.close()
