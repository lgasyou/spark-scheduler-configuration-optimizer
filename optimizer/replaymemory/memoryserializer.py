import pickle

from . import ReplayMemoryProxy
from ..util import fileutil


class MemorySerializer(object):

    FINAL_SAVE_FILENAME = './results/pre-train-replay-memory.pk'

    def __init__(self, proxy: ReplayMemoryProxy):
        self.mem = proxy.memory

    def try_load(self, filename: str = None):
        """
        Load from a file.
        :return: Whether load succeed.
        """
        filename = filename or self.FINAL_SAVE_FILENAME
        mem = self.mem

        if not fileutil.file_exists(filename):
            print('File: ', filename, "doesn't exist.")
            return False

        with open(filename, 'rb') as f:
            mem.t, mem.transitions = pickle.load(f)
            return True

    def save(self, filename: str = None):
        """
        Save into a file.
        """
        filename = filename or self.FINAL_SAVE_FILENAME
        mem = self.mem

        with open(filename, 'wb') as f:
            pickle.dump([mem.t, mem.transitions], f)
