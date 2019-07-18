import logging
import time

from optimizer.agent import Agent
from optimizer.environment import PreTrainEnv, StateInvalidException
from optimizer.hyperparameters import PRE_TRAIN_LOOP_INTERNAL
from optimizer.replaymemory import ReplayMemoryProxy
from optimizer.replaymemory.memoryserializer import MemorySerializer


class PreTrainer(object):

    TMP_MEMORY_FILENAME = './results/pre-train-memory.tmp.pk'

    def __init__(self, args, memory: ReplayMemoryProxy, agent: Agent, action_space: int):
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.mem = memory
        self.memory_serializer = MemorySerializer(memory)
        self.agent = agent
        self.env = PreTrainEnv(self.args)
        self.action_space = action_space

    def start_from_breakpoint(self):
        self.logger.info('Pre-training DQN model...')

        if not self.memory_serializer.try_load():
            if not self.memory_serializer.try_load_by_filename(self.TMP_MEMORY_FILENAME):
                while True:
                    self._train_step()

            self.memory_serializer.save()

        # Pre-train DQN model by using training set
        self.agent.learn(self.mem)
        self.logger.info('Pre-training DQN model finished.')

    def _train_step(self):
        self.env.generate_pre_train_set()
        pre_train_set = self.env.generate_pre_train_set()
        for action_index in range(self.action_space):
            self.env.start(action_index, pre_train_set)
            self._train_once(action_index)
            self.memory_serializer.save_as(self.TMP_MEMORY_FILENAME)

    def _train_once(self, action_index: int):
        T, done = 0, False
        while not done:
            try:
                state, reward, done = self.env.step()
                self.logger.info('Iteration: %d, Action: %d, Reward: %f' % (T, action_index, reward))
                self.mem.append(state, action_index, reward, done)
                T += 1
            except StateInvalidException:
                pass
            time.sleep(PRE_TRAIN_LOOP_INTERNAL)

        self.mem.terminate()
