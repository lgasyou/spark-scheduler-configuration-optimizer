import argparse
import time

from optimizer.controller.abstractcontroller import AbstractController
from optimizer.environment import Env
from optimizer.hyperparameters import OPTIMIZATION_LOOP_INTERNAL


class OptimizationController(AbstractController):

    def run(self):
        self._load_memory()
        interval = OPTIMIZATION_LOOP_INTERNAL
        self.env.reset_buffer()
        self.logger.info('Started optimizing cluster.')
        state = self.env.try_get_state()
        while True:
            state, action, reward, done = self.optimize_timestep(state, self.agent.act)
            self.logger.info("Episode {}: Reward {}, Action {}, Done {}".format(self.t, reward, action, done))
            time.sleep(interval)

    def _env(self, args: argparse.Namespace):
        return Env(args)
