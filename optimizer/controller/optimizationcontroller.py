import argparse
import time

from optimizer.controller.abstractcontroller import AbstractController
from optimizer.environment import Env, StateInvalidException
from optimizer.hyperparameters import TRAIN_LOOP_INTERNAL


class OptimizationController(AbstractController):

    def run(self):
        env = self.env
        args = self.args
        dqn = self.agent
        mem = self.mem

        priority_weight_increase = self.priority_weight_increase
        reward_clip = args.reward_clip

        T, done, next_state = 0, False, None
        env.reset_buffer()
        state = env.get_state()

        while True:
            action = dqn.act(state)

            try:
                next_state, reward, done = env.step(action)  # Step
                if reward_clip > 0:
                    reward = max(min(reward, reward_clip), -reward_clip)  # Clip rewards
                print(reward, action)
                mem.append(state, action, reward, done)  # Append transition to memory
                time.sleep(TRAIN_LOOP_INTERNAL)
                T += 1
            except StateInvalidException as e:
                print(e)

            if done:
                break

            if T % args.replay_frequency == 0:
                dqn.reset_noise()  # Draw a new set of noisy weights

            # Train and test
            if T >= args.learn_start:
                # Anneal importance sampling weight β to 1
                mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

                if T % args.replay_frequency == 0:
                    dqn.learn(mem)  # Train with n-step distributional double-Q learning

                # Update target network
                if T % args.target_update == 0:
                    dqn.update_target_net()

            state = next_state

    def _env(self, args: argparse.Namespace):
        return Env(args)
