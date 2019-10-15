import time
import logging

from optimizer.controller.abstractcontroller import AbstractController
from optimizer.environment import TrainingEnv
from optimizer.replaymemory.memoryserializer import MemorySerializer
from optimizer.hyperparameters import EXTRA_WAIT_TIME


class TrainingController(AbstractController):

    TMP_MEMORY_FILENAME = './results/pre-train-memory.tmp.pk'

    def __init__(self, args):
        super().__init__(args)
        self.memory_serializer = MemorySerializer(self.mem)

    def run(self):
        self.logger.info('Starting training DRL agent...')
        self.start_from_breakpoint()
        self.logger.info('Agent training finished.')

    def start_from_breakpoint(self):
        if not self._load_memory():
            self.memory_serializer.try_load_by_filename(self.TMP_MEMORY_FILENAME)
            self.t = self.mem.index
            self.logger.info('Start from episode %d.' % self.t)
            while True:
                self._train_step()

    def _train_step(self):
        self.env.reset()
        self._train_episode()
        self._save_progress()

    def _train_episode(self):
        done, interval = False, EXTRA_WAIT_TIME
        state = self.env.try_get_state()
        while not done:
            state, action, reward, done = self.optimize_timestep(state, self.agent.act)
            if self.t % 100 == 0:
                self.logger.info("Time Step {}: Reward {}, Action {}, Done {}".format(self.t, reward, action, done))
            if not self.simulating:
                time.sleep(interval)

    def _save_progress(self):
        self.memory_serializer.save_as(self.TMP_MEMORY_FILENAME)
        self._save_agent()

    def _save_agent(self):
        self.agent.save(directory='results')

    def _env(self, args):
        return TrainingEnv(self.args)
