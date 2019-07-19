import logging
import pickle

from optimizer.replaymemory import ReplayMemoryProxy
from optimizer.util import fileutil


class MemorySerializer(object):

    FINAL_SAVE_FILENAME = './results/pre-train-replay-memory.pk'

    def __init__(self, proxy: ReplayMemoryProxy):
        self.logger = logging.getLogger(__name__)
        self.mem = proxy.memory

    def try_load(self):
        """
        Load from file.
        :return: Whether load succeed.
        """
        return self.try_load_by_filename(self.FINAL_SAVE_FILENAME)

    def try_load_by_filename(self, filename):
        mem = self.mem

        if not fileutil.file_exists(filename):
            self.logger.info("File: %s doesn't exist." % filename)
            return False

        with open(filename, 'rb') as f:
            mem.t, mem.transitions = pickle.load(f)
            return True

    def save(self):
        self.save_as(self.FINAL_SAVE_FILENAME)

    def save_as(self, filename: str):
        mem = self.mem
        with open(filename, 'wb') as f:
            pickle.dump([mem.t, mem.transitions], f)
            self.logger.info('File %s saved.' % filename)
