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
            while True:
                self._train_step()

        # Pre-train DQN model by using training set
        self.agent.learn(self.mem)

    def _train_step(self):
        pre_train_set = self.env.generate_pre_train_set()
        for action_index in range(self.action_space):
            self.env.start(action_index, pre_train_set)
            self._train_episode(action_index)
            self._save_progress()
            sparkutil.clean_spark_log(os.getcwd(), self.args.hadoop_home)

    def _train_episode(self, action_index: int):
        done, interval = False, TRAINING_LOOP_INTERNAL
        while not done:
            self._reset_noise()

            state, reward, done = self.env.step(interval)
            reward = self._clip_reward(reward)
            self.mem.append(state, action_index, reward, done)

            if done:
                self.mem.terminate()
            self.t += 1

            self.logger.info("Episode {}: Action {}, Reward {}, Done {}".format(self.t, action_index, reward, done))

            # Train
            if self.t >= self.args.learn_start:
                self._agent_learn()

            time.sleep(interval)

    def _save_progress(self):
        self.memory_serializer.save_as(self.TMP_MEMORY_FILENAME)

    def _env(self, args):
        return TrainingEnv(self.args)
