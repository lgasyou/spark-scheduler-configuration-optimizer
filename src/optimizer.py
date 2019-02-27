import argparse
import random
from datetime import datetime

import numpy as np
import torch

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


class Optimizer(object):

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
            if done:
                state, done = self.env.reset(), False
                # Anneal importance sampling weight β to 1
                self.mem.priority_weight = min(self.mem.priority_weight + self.priority_weight_increase, 1)
                if T % self.args.replay_frequency == 0:
                    self.dqn.learn(self.mem)  # Train with n-step distributional double-Q learning
                    self.dqn.eval()  # Set DQN (online network) to evaluation mode
                    avg_reward, avg_Q = test(self.args, 0, self.dqn, val_mem, evaluate=True)  # Test

                # Update target network
                if T % self.args.target_update == 0:
                    self.dqn.update_target_net()
            state = next_state

        self.env.close()
