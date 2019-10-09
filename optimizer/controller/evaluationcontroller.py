import argparse
import time
import logging

from optimizer.controller.abstractcontroller import AbstractController
from optimizer.environment import EvaluationEnv
from optimizer.environment.stateobtaining.regulardelayfetcher import RegularDelayFetcher
from optimizer.hyperparameters import EVALUATION_LOOP_INTERNAL


class EvaluationController(AbstractController):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        if not self.simulating:
            self.delay_fetcher = RegularDelayFetcher(self.env.communicator.state_builder)
        self.costs = []
        self.episode = 0

    def run(self):
        self.logger.info('Evaluating with optimization.')
        self.run_with_optimization()

        self.logger.info('Evaluating without optimization.')
        for action_index in range(self.action_space):
            self.run_without_optimization(action_index)

    def run_with_optimization(self):
        self.costs.clear()
        for i in range(self.args.evaluation_episodes):
            self.episode = i
            self.env.reset()
            if not self.simulating:
                self.delay_fetcher.start_heartbeat()
            self.with_optimize_episode()
            self.cleanup(
                time_costs_filename='./results/optim-time-costs-%d.xlsx' % i,
                delays_filename='./results/optim-delays-%d.txt' % i
            )

    def with_optimize_episode(self):
        interval = EVALUATION_LOOP_INTERNAL
        self.env.reset_buffer()
        done, state = False, self.env.try_get_state()
        while not done:
            state, action, reward, done = self.optimize_timestep(state, self.agent.act_e_greedy)

            if self.simulating and self.t % 20 == 0:
                cost = self.env.get_total_time_cost()
                self.costs.append({self.t: cost})
                self.logger.info('Episode: {}, Time Cost: {}'.format(self.episode, cost))

            self.logger.info("Episode {}, Time {}: Reward {}, Action {}, Done {}"
                             .format(self.episode, self.t, reward, action, done))
            if not self.simulating:
                time.sleep(interval)

    def run_without_optimization(self, action_index: int):
        self.costs.clear()
        for i in range(self.args.evaluation_episodes):
            self.episode = i
            self.env.reset()
            if not self.simulating:
                self.delay_fetcher.start_heartbeat()
            self.without_optimize_episode(action_index)
            self.cleanup(
                time_costs_filename='./results/no-optim-time-costs-%d-%d.xlsx' % (action_index, i),
                delays_filename='./results/no-optim-delays-%d-%d.txt' % (action_index, i)
            )

    def without_optimize_episode(self, action_index: int):
        interval = EVALUATION_LOOP_INTERNAL
        self.env.reset_buffer()
        done, state = False, self.env.try_get_state()
        self.t = 0
        while not done:
            _, reward, done = self.env.step(action_index)

            if self.simulating and self.t % 20 == 0:
                cost = self.env.get_total_time_cost()
                self.costs.append({self.t: cost})
                self.logger.info('Episode: {}, Time Cost: {}'.format(self.episode, cost))

            self.logger.info("Episode {}: Reward {}, Action {}, Done {}"
                             .format(self.episode, reward, action_index, done))
            self.t += 1
            if not self.simulating:
                time.sleep(interval)

    def cleanup(self, time_costs_filename: str, delays_filename: str):
        if not self.simulating:
            self.delay_fetcher.save_delays(delays_filename)
            self.delay_fetcher.stop()

        cost = self.env.get_total_time_cost()
        self.costs.append({self.t: cost})
        logging.info(self.costs)
        with open(delays_filename, 'w') as f:
            print(self.costs, file=f)
        self.logger.info('Summary %s saved.' % delays_filename)

    def _env(self, args: argparse.Namespace):
        return EvaluationEnv(args)
