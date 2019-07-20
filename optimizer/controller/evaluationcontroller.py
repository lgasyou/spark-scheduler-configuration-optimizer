import argparse
import os
import time

from optimizer.controller.abstractcontroller import AbstractController
from optimizer.environment import EvaluationEnv
from optimizer.hyperparameters import EVALUATION_LOOP_INTERNAL
from optimizer.util import excelutil, sparkutil


class EvaluationController(AbstractController):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.costs = []
        self.episode = 0

    def run(self):
        self.logger.info('Running without optimization.')
        for action_index in range(self.action_space):
            self.run_without_optimization(action_index)

        # self.logger.info('Running with optimization.')
        # self.run_with_optimization()

    def run_with_optimization(self):
        self._load_memory()
        self.agent.train()
        self.agent.learn(self.mem)
        self.logger.info('Agent learnt.')
        self.agent.eval()

        self.costs.clear()
        for i in range(self.args.evaluation_episodes):
            self.episode = i
            self.env.reset()
            self.with_optimize_episode()
            self.cleanup('./results/optimization-%d.xlsx' % i)

    def with_optimize_episode(self):
        interval = EVALUATION_LOOP_INTERNAL
        self.env.reset_buffer()
        done, state = False, self.env.try_get_state(interval)
        while not done:
            state, action, reward, done = self.optimize_episode(state, self.agent.act_e_greedy, interval)
            self.logger.info("Episode {}, Time {}: Reward {}, Action {}, Done {}"
                             .format(self.episode, self.t, reward, action, done))
            time.sleep(interval)

        self.mem.terminate()

    def run_without_optimization(self, action_index: int):
        self.costs.clear()
        for i in range(self.args.evaluation_episodes):
            self.episode = i
            self.env.reset()
            self.without_optimize_episode(action_index)
            self.cleanup('./results/no-optimization-%d-%d.xlsx' % (action_index, i))

    def without_optimize_episode(self, action_index: int):
        interval = EVALUATION_LOOP_INTERNAL
        self.env.reset_buffer()
        done, state = False, self.env.try_get_state(interval)
        while not done:
            _, reward, done = self.env.step(action_index, interval)
            self.logger.info("Episode {}: Reward {}, Action {}, Done {}"
                             .format(self.episode, reward, action_index, done))
            time.sleep(interval)

    def cleanup(self, save_filename: str):
        costs, _ = self.env.get_total_time_cost()
        self.costs.append(costs)
        self.logger.info('Episode: {}, Time Cost: {}'.format(self.episode, costs))
        excelutil.list2excel(self.costs, save_filename)
        self.logger.info('Summary {} saved.' % save_filename)
        sparkutil.clean_spark_log(os.getcwd(), self.args.hadoop_home)

    # def run_with_optimization(self):
    #     agent.eval()  # Set DQN (online network) to evaluation mode
    #     avg_reward, avg_Q, time_cost = self.validator.evaluate(T)  # Test
    #     self.logger.info('T = ' + str(T) + ' / ' + str(num_training_steps) +
    #                      ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q) +
    #                      ' | Total Time Cost: ' + str(time_cost) + 'ms')
    #     agent.train()  # Set DQN (online network) back to training mode

    def _env(self, args: argparse.Namespace):
        return EvaluationEnv(args)
