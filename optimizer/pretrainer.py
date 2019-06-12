import logging
import time

from .agent import Agent
from .environment import PreTrainEnv, StateInvalidException
from .hyperparameters import PRE_TRAIN_LOOP_INTERNAL
from .replaymemory import ReplayMemory
from .replaymemory.memoryserializer import MemorySerializer
from .util import fileutil


class PreTrainer(object):

    MARK_FILENAME = './results/pre-train-mark'
    MEMORY_FILENAME_TEMPLATE = './results/pre-train-replay-memory-%d-%d.pk'

    def __init__(self, memory: ReplayMemory, agent: Agent,  args):
        self.mem = memory
        self.memory_serializer = MemorySerializer(memory)
        self.dqn = agent
        self.args = args
        self.logger = logging.getLogger(__name__)

    def start_pre_train(self):
        self.logger.info('Pre-training DQN model...')

        # Try to load data from file, if fails run training set and
        # save them into memory.
        if not self.memory_serializer.try_load():
            for action_index, file_index in self._train_range():
                self.train_once(action_index, file_index)
                self.memory_serializer.save(self.MEMORY_FILENAME_TEMPLATE % (action_index, file_index))
                self._mark(action_index, file_index)

            # Save data
            self.memory_serializer.save()

        # Pre-train DQN model by using training set
        self.dqn.learn(self.mem)
        self.logger.info('Pre-training DQN model finished.')

    def start_from_breakpoint(self):
        pass

    def train_once(self, action_index: int, file_index: int):
        train_env = PreTrainEnv(self.args)
        train_env.start_sls(file_index, action_index)

        T, done = 0, False
        while not done:
            try:
                state, reward, done = train_env.step()
                print('Iteration: %d, Action: %d, File: %d, Reward: %f' % (T, action_index, file_index, reward))
                self.mem.append(state, action_index, reward, done)
                time.sleep(PRE_TRAIN_LOOP_INTERNAL)
                T += 1
            except StateInvalidException:
                done = True

    def _mark(self, action_index: int, file_index: int):
        with open(self.MARK_FILENAME) as f:
            f.write('%d,%d' % (action_index, file_index))

    def _get_mark(self):
        if fileutil.file_exists(self.MARK_FILENAME):
            with open(self.MARK_FILENAME) as f:
                data = f.readline().split(',')
                return data[0], data[1]
        return -1, -1

    @staticmethod
    def _train_range():
        for action_index in [2]:
            for hour in range(24):
                yield action_index, hour
