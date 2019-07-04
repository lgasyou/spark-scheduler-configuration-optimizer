import argparse
import logging
import os
import random
import time

import plotly
import torch
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line

from ..environment import EvaluationEnv, StateInvalidException
from ..hyperparameters import TEST_LOOP_INTERNAL, EVALUATION_LOOP_INTERNAL
from ..replaymemory import MemorySerializer, ReplayMemoryProxy
from optimizer.util import excelutil

# Globals
Ts, rewards, Qs, best_avg_reward = [], [], [], -1e10


class Validator(object):

    def __init__(self, args: argparse.Namespace, env: EvaluationEnv, agent, action_space: int):
        self.args = args
        self.env = env
        self.agent = agent
        self.action_space = action_space
        self.evaluate_cnt = 0
        self.logger = logging.getLogger(__name__)
        self.val_mem = self._setup_val_mem()

    def evaluate(self, T, evaluate=False):
        args = self.args
        dqn = self.agent
        val_mem = self.val_mem

        global Ts, rewards, Qs, best_avg_reward
        env = EvaluationEnv(args)
        env.eval()
        Ts.append(T)
        T_rewards, T_Qs = [], []
        total_time_cost_ms = 0
        arr = []

        # Test performance over several episodes
        done, reward_sum, state = True, 0, None
        for i in range(args.evaluation_episodes):
            while True:
                self.logger.info('Evaluation Loop %d' % i)
                if done:
                    state, reward_sum, done = env.reset(), 0, False

                action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
                print("Action:", action)
                try:
                    state, reward, done = env.step(action)  # Step
                    print('Reward:', reward)
                    reward_sum += reward
                    time.sleep(TEST_LOOP_INTERNAL)
                except StateInvalidException:
                    done = True

                if done:
                    T_rewards.append(reward_sum)
                    time.sleep(EVALUATION_LOOP_INTERNAL)
                    costs, time_cost_ms = env.get_total_time_cost()
                    print('Iteration', i, 'Time Cost:', costs)
                    arr.append(list(costs))

                    total_time_cost_ms += time_cost_ms
                    break

        print('Total Time Cost :', total_time_cost_ms, 'ms')
        print(arr)
        excelutil.list2excel(arr, './results/evaluate_%d.xlsx' % self.evaluate_cnt)
        self.evaluate_cnt += 1
        env.close()

        # Test Q-values over validation memory
        for state in val_mem:  # Iterate over valid states
            T_Qs.append(dqn.evaluate_q(state))

        avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
        if not evaluate:
            # Append to results
            rewards.append(T_rewards)
            Qs.append(T_Qs)

            # Plot
            self._plot_line(Ts, rewards, 'Reward', path='results')
            self._plot_line(Ts, Qs, 'Q', path='results')

            # Save model parameters if improved
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                dqn.save('results')

        # Return average reward and Q-value
        return avg_reward, avg_Q, total_time_cost_ms

    # Plots min, max and mean + standard deviation bars of a population over time
    @staticmethod
    def _plot_line(xs, ys_population, title, path=''):
        max_colour, mean_colour, std_colour, transparent = \
            'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

        ys = torch.tensor(ys_population, dtype=torch.float32)
        ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(
            1).squeeze()
        ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

        trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
        trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.',
                              showlegend=False)
        trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour,
                             line=Line(color=mean_colour),
                             name='Mean')
        trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour,
                              line=Line(color=transparent),
                              name='-1 Std. Dev.', showlegend=False)
        trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

        plotly.offline.plot({
            'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
            'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
        }, filename=os.path.join(path, title + '.html'), auto_open=False)

    # Construct validation memory
    def _setup_val_mem(self) -> ReplayMemoryProxy:
        self.logger.info('Setting up validation memory...')
        val_mem = ReplayMemoryProxy(self.args, self.args.evaluation_size)
        memory_serializer = MemorySerializer(val_mem)

        if memory_serializer.try_load_by_filename('./results/validation-replay-memory.pk'):
            self.logger.info('Validation memory setting up finished.')
            return val_mem

        # T, done, state = 0, True, None
        # while T < self.args.evaluation_size:
        #     self.logger.info('Validation Loop %d' % T)
        #     if done:
        #         state, done = self.env.reset(), False
        #
        #     try:
        #         next_state, reward, done = self.env.step(random.randint(0, self.action_space - 1))
        #         print('Reward: %f' % reward)
        #         time.sleep(EVALUATION_LOOP_INTERNAL)
        #     except StateInvalidException:
        #         val_mem.terminate()
        #         done = True
        #         continue
        #
        #     val_mem.append(state, None, None, done)
        #     state = next_state
        #     T += 1
        # memory_serializer.save('./results/validation-replay-memory.pk')

        self.logger.info('Validation memory setting up finished.')
        return val_mem
