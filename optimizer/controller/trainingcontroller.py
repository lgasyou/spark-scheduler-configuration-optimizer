import os
import time

from optimizer.controller.abstractcontroller import AbstractController
from optimizer.environment import TrainingEnv
from optimizer.hyperparameters import TRAINING_LOOP_INTERNAL
from optimizer.replaymemory.memoryserializer import MemorySerializer
from optimizer.util import sparkutil


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
            try:
                while True:
                    self._train_step()
            except InterruptedError:
                self._save_agent()
                raise InterruptedError

    def _train_step(self):
        pre_train_set = self.env.generate_pre_train_set()
        self.env.start(pre_train_set)
        self._train_episode()
        self._save_progress()
        sparkutil.clean_spark_log(os.getcwd(), self.args.hadoop_home)

    def _train_episode(self):
        done, interval = False, TRAINING_LOOP_INTERNAL
        state = self.env.try_get_state(interval)
        while not done:
            state, action, reward, done = self.optimize_episode(state, self.agent.act)
            self.logger.info("Episode {}: Reward {}, Action {}, Done {}".format(self.t, reward, action, done))
            time.sleep(interval)

    def _save_progress(self):
        self.memory_serializer.save_as(self.TMP_MEMORY_FILENAME)
        self._save_agent()

    def _save_agent(self):
        self.agent.save('./results')

    def _env(self, args):
        return TrainingEnv(self.args)
