import argparse
import logging
import random
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from .env import EvaluationEnv
from .test import test
from ..controller import AbstractController
from ..environment.exceptions import StateInvalidException
from ..memory import ReplayMemory


class EvaluationController(AbstractController):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.logger = logging.getLogger(__name__)
        self.val_mem = self._setup_val_mem()

    def run(self):
        self.logger.info('Running with optimization.')
        self.run_with_optimization()

        self.logger.info('Running without optimization.')
        self.run_without_optimization()

    def run_with_optimization(self):
        args = self.args
        dqn = self.dqn
        env = self.env
        mem = self.mem
        val_mem = self.val_mem

        priority_weight_increase = self.priority_weight_increase
        num_training_steps = args.T_max
        reward_clip = args.reward_clip

        T, done, state = 0, True, None
        dqn.train()
        for T in tqdm(range(num_training_steps)):
            if done:
                state, done = env.reset(), False

            if T % args.replay_frequency == 0:
                dqn.reset_noise()  # Draw a new set of noisy weights

            action = dqn.act(state)  # Choose an action greedily (with noisy weights)
            try:
                next_state, reward, done = env.step(action)  # Step
                if reward_clip > 0:
                    reward = max(min(reward, reward_clip), -reward_clip)  # Clip rewards
                print('Reward:', reward)
                mem.append(state, action, reward, done)  # Append transition to memory
                time.sleep(1)
                T += 1
            except StateInvalidException:
                done = True
                continue

            # Train and test
            if T >= args.learn_start:
                # Anneal importance sampling weight β to 1
                mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

                if T % args.replay_frequency == 0:
                    dqn.learn(mem)  # Train with n-step distributional double-Q learning

                if T % args.evaluation_interval == 0:
                    # Shutdown current SLS runner so that we can start another in test process.
                    env.close()
                    done = True

                    dqn.eval()  # Set DQN (online network) to evaluation mode
                    avg_reward, avg_Q, time_cost = test(args, T, dqn, val_mem)  # Test
                    self.logger.info('T = ' + str(T) + ' / ' + str(num_training_steps) +
                                     ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q) +
                                     ' | Total Time Cost: ' + str(time_cost) + 'ms')
                    dqn.train()  # Set DQN (online network) back to training mode

                # Update target network
                if T % args.target_update == 0:
                    dqn.update_target_net()
            state = next_state

        env.close()

    def run_without_optimization(self, action_index=0):
        env = EvaluationEnv(self.args)
        env.eval()
        total_time_cost_ms = 0
        time_costs = np.zeros(shape=(self.args.evaluation_episodes, 31), dtype=int)

        # Test performance over several episodes
        done, reward_sum, state = True, 0, None
        for T in range(self.args.evaluation_episodes):
            while True:
                self.logger.info('Evaluation Loop %d' % T)
                if done:
                    state, reward_sum, done = env.reset(), 0, False

                try:
                    state, reward, done = env.step(action_index)  # Step
                    print('Reward:', reward)
                    reward_sum += reward
                    time.sleep(5)
                except StateInvalidException:
                    done = True

                if done:
                    time.sleep(30)
                    costs, time_cost_ms = env.get_total_time_cost()
                    print('Iteration', T, 'Time Cost:', costs)
                    time_costs[T] = list(costs)
                    total_time_cost_ms += time_cost_ms
                    break

        print('Total Time Cost :', total_time_cost_ms, 'ms')
        pd.DataFrame(time_costs).to_csv('./results/time_costs.csv')
        env.close()

    def _env(self, args: argparse.Namespace):
        return EvaluationEnv(args)

    # Construct validation memory
    def _setup_val_mem(self) -> ReplayMemory:
        self.logger.info('Setting up validation memory...')
        val_mem = ReplayMemory(self.args, self.args.evaluation_size)

        if val_mem.try_load_from_file('./results/validation-replay-memory.pk'):
            self.logger.info('Validation memory setting up finished.')
            return val_mem

        T, done, state = 0, True, None
        while T < self.args.evaluation_size:
            self.logger.info('Validation Loop %d' % T)
            if done:
                state, done = self.env.reset(), False

            try:
                next_state, _, done = self.env.step(random.randint(0, self.action_space - 1))
                time.sleep(5)
            except StateInvalidException:
                done = True
                continue

            val_mem.append(state, None, None, done)
            state = next_state
            T += 1
        val_mem.save('./results/validation-replay-memory.pk')

        self.logger.info('Validation memory setting up finished.')
        return val_mem
