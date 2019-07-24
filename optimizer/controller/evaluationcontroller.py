import argparse
import os
import time

from optimizer.controller.abstractcontroller import AbstractController
from optimizer.environment import EvaluationEnv
from optimizer.environment.stateobtaining.regulartimedelayfetcher import RegularTimeDelayFetcher
from optimizer.hyperparameters import EVALUATION_LOOP_INTERNAL
from optimizer.util import excelutil, sparkutil


class EvaluationController(AbstractController):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.time_delay_fetcher = RegularTimeDelayFetcher(self.env.communicator.state_builder)
        self.costs = []
        self.episode = 0

    def run(self):
        self.logger.info('Evaluating with optimization.')
        self.run_with_optimization()

        self.logger.info('Evaluating without optimization.')
        for action_index in [2, 5, 8]:
            self.run_without_optimization(action_index)

    def run_with_optimization(self):
        self._load_memory()
        self.agent.train()
        self.agent.learn(self.mem)
        self.agent.eval()

        self.costs.clear()
        for i in range(self.args.evaluation_episodes):
            self.episode = i
            self.env.reset()
            self.time_delay_fetcher.start_heartbeat()
            self.with_optimize_episode()
            self.cleanup(
                time_costs_filename='./results/optim-time-costs-%d.xlsx' % i,
                time_delays_filename='./results/optim-time-delays-%d.txt' % i
            )

    def with_optimize_episode(self):
        interval = EVALUATION_LOOP_INTERNAL
        self.env.reset_buffer()
        done, state = False, self.env.try_get_state()
        while not done:
            state, action, reward, done = self.optimize_episode(state, self.agent.act_e_greedy)
            self.logger.info("Episode {}, Time {}: Reward {}, Action {}, Done {}"
                             .format(self.episode, self.t, reward, action, done))
            time.sleep(interval)

    def run_without_optimization(self, action_index: int):
        self.costs.clear()
        for i in range(self.args.evaluation_episodes):
            self.episode = i
            self.env.reset()
            self.time_delay_fetcher.start_heartbeat()
            self.without_optimize_episode(action_index)
            self.cleanup(
                time_costs_filename='./results/no-optim-time-costs-%d-%d.xlsx' % (action_index, i),
                time_delays_filename='./results/no-optim-time-delays-%d-%d.txt' % (action_index, i)
            )

    def without_optimize_episode(self, action_index: int):
        interval = EVALUATION_LOOP_INTERNAL
        self.env.reset_buffer()
        done, state = False, self.env.try_get_state()
        while not done:
            _, reward, done = self.env.step(action_index)
            self.logger.info("Episode {}: Reward {}, Action {}, Done {}"
                             .format(self.episode, reward, action_index, done))
            time.sleep(interval)

    def cleanup(self, time_costs_filename: str, time_delays_filename: str):
        self.time_delay_fetcher.save_time_delays(time_delays_filename)
        self.time_delay_fetcher.stop()

        costs, _ = self.env.get_total_time_cost()
        self.costs.append(costs)
        self.logger.info('Episode: {}, Time Cost: {}'.format(self.episode, costs))
        excelutil.list2excel(self.costs, time_costs_filename)
        self.logger.info('Summary %s saved.' % time_costs_filename)

        sparkutil.clean_spark_log(os.getcwd(), self.args.hadoop_home)

    def _env(self, args: argparse.Namespace):
        return EvaluationEnv(args)
