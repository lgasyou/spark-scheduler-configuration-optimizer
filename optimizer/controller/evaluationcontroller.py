import argparse
import logging
import time

from tqdm import tqdm

from .abstractcontroller import AbstractController
from .validator import Validator
from ..environment import EvaluationEnv, StateInvalidException
from ..hyperparameters import TRAIN_LOOP_INTERNAL, EVALUATION_LOOP_INTERNAL


class EvaluationController(AbstractController):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.logger = logging.getLogger(__name__)
        self.validator = Validator(args, self.env, self.agent, self.action_space)

    def run(self):
        self.logger.info('Running with optimization.')
        self.run_with_optimization()

        self.logger.info('Running without optimization.')
        self.run_without_optimization()

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
            if done:
                state, done = env.reset(), False

            if T % args.replay_frequency == 0:
                agent.reset_noise()  # Draw a new set of noisy weights

            action = agent.act(state)  # Choose an action greedily (with noisy weights)
            try:
                next_state, reward, done = env.step(action)  # Step
                if reward_clip > 0:
                    reward = max(min(reward, reward_clip), -reward_clip)  # Clip rewards
                print('Reward:', reward)
                mem.append(state, action, reward, done)  # Append transition to memory
                time.sleep(TRAIN_LOOP_INTERNAL)
                T += 1
            except StateInvalidException:
                done = True
                mem.terminate()
                continue

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

    def run_without_optimization(self, action_index=2):
        env = EvaluationEnv(self.args)
        env.eval()
        total_time_cost_ms = 0

        arr = []
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
                    time.sleep(EVALUATION_LOOP_INTERNAL)
                except StateInvalidException:
                    done = True

                if done:
                    time.sleep(EVALUATION_LOOP_INTERNAL)
                    costs, time_cost_ms = env.get_total_time_cost()
                    print('Iteration', T, 'Time Cost:', costs)
                    arr.append(costs)
                    total_time_cost_ms += time_cost_ms
                    break

        print('Total Time Cost :', total_time_cost_ms, 'ms')
        print(arr)
        env.close()

    def _env(self, args: argparse.Namespace):
        return EvaluationEnv(args)
