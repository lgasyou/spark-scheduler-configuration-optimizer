import argparse
import time

from optimizer.controller.abstractcontroller import AbstractController
from optimizer.environment import Env
from optimizer.hyperparameters import TRAIN_LOOP_INTERNAL


class OptimizationController(AbstractController):

    def run(self):
        interval = TRAIN_LOOP_INTERNAL
        self.env.reset_buffer()
        state = self.env.try_get_state(interval)
        while True:
            state, reward, done = self.optimize_episode(state, self.agent.act, interval)
            self.logger.info("Episode {}: Reward {}, Done {}".format(self.t, reward, done))
            time.sleep(interval)

    def _env(self, args: argparse.Namespace):
        return Env(args)
