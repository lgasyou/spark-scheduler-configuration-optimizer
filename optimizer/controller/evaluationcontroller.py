import argparse
import logging
import os
import time

from tqdm import tqdm

from optimizer.controller.abstractcontroller import AbstractController
from optimizer.controller.validator import Validator
from optimizer.environment import EvaluationEnv, StateInvalidException
from optimizer.hyperparameters import TRAIN_LOOP_INTERNAL, EVALUATION_LOOP_INTERNAL
from optimizer.util import excelutil, processutil


class EvaluationController(AbstractController):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.logger = logging.getLogger(__name__)
        self.validator = Validator(args, self.env, self.agent, self.action_space)

    def run(self):
        self.logger.info('Running without optimization.')
        for action_index in range(self.action_space):
            self.run_without_optimization(action_index)

        # self.logger.info('Running with optimization.')
        # self.run_with_optimization()

    # TODO: Refactor this code.
    def run_with_optimization(self):
        args = self.args
        agent = self.agent
        env = self.env
        mem = self.mem

        priority_weight_increase = self.priority_weight_increase
        num_training_steps = args.T_max
        reward_clip = args.reward_clip

        T, done, state = 0, True, None
        agent.train()
        for T in tqdm(range(num_training_steps)):
            try:
                if done:
                    state, done = env.reset(), False
                    mem.terminate()

                if T % args.replay_frequency == 0:
                    agent.reset_noise()  # Draw a new set of noisy weights

                action = agent.act(state)  # Choose an action greedily (with noisy weights)
                next_state, reward, done = env.step(action)  # Step
                if reward_clip > 0:
                    reward = max(min(reward, reward_clip), -reward_clip)  # Clip rewards
                self.logger.info('Reward: %f', reward)
                mem.append(state, action, reward, done)  # Append transition to memory
                T += 1
            except StateInvalidException:
                if done:
                    done = False
            finally:
                time.sleep(TRAIN_LOOP_INTERNAL)

            # Train and test
            if T >= args.learn_start:
                # Anneal importance sampling weight β to 1
                mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

                if T % args.replay_frequency == 0:
                    agent.learn(mem)  # Train with n-step distributional double-Q learning

                if T % args.evaluation_interval == 0:
                    # Shutdown current SLS runner so that we can start another in test process.
                    env.close()
                    done = True

                    agent.eval()  # Set DQN (online network) to evaluation mode
                    avg_reward, avg_Q, time_cost = self.validator.evaluate(T)  # Test
                    self.logger.info('T = ' + str(T) + ' / ' + str(num_training_steps) +
                                     ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q) +
                                     ' | Total Time Cost: ' + str(time_cost) + 'ms')
                    agent.train()  # Set DQN (online network) back to training mode

                # Update target network
                if T % args.target_update == 0:
                    agent.update_target_net()
            state = next_state

        env.close()

    def run_without_optimization(self, action_index):
        env = EvaluationEnv(self.args)
        env.eval()
        total_time_cost_ms = 0

        arr = []
        # Test performance over several episodes
        done = True
        for T in range(self.args.evaluation_episodes):
            while True:
                self.logger.info('Evaluation Loop %d' % T)
                try:
                    if done:
                        _, done = env.reset(), False

                    _, reward, done = env.step(action_index)  # Step
                    self.logger.info('Reward: %f', reward)
                except StateInvalidException:
                    done = False if done is True else False
                time.sleep(EVALUATION_LOOP_INTERNAL)

                if done:
                    time.sleep(EVALUATION_LOOP_INTERNAL)
                    costs, time_cost_ms = env.get_total_time_cost()
                    arr.append(costs)
                    total_time_cost_ms += time_cost_ms
                    self.logger.info('Iteration: {}, Time Cost: {}'.format(T, costs))
                    excelutil.list2excel(arr, './results/no-optimization-%d-%d.xlsx' % (action_index, T))
                    clean_spark_log(os.getcwd(), self.args.hadoop_home)
                    break

        self.logger.info('Total Time Cost :%d ms', total_time_cost_ms)

        excelutil.list2excel(arr, './results/no-optimization-%d.xlsx' % action_index)
        env.close()

    def _env(self, args: argparse.Namespace):
        return EvaluationEnv(args)


def clean_spark_log(wd, hadoop_home):
    cmd = ['%s/bin/clean-spark-log.sh' % wd, hadoop_home]
    return processutil.start_process(cmd)
