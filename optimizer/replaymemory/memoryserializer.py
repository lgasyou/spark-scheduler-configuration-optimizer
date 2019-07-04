import pickle

from . import ReplayMemoryProxy
from ..util import fileutil


class MemorySerializer(object):

    FINAL_SAVE_FILENAME = './results/pre-train-replay-memory.pk'
    MEMORY_FILENAME_TEMPLATE = './results/pre-train-replay-memory-%d-%d.pk'

    def __init__(self, proxy: ReplayMemoryProxy):
        self.mem = proxy.memory

    def try_load(self, action_index=-1, file_index=-1):
        """
        Load from a file.
        :return: Whether load succeed.
        """
        if action_index == -1 or file_index == -1:
            filename = self.FINAL_SAVE_FILENAME
        else:
            filename = self._get_serialized_memory_filename(action_index, file_index)

        return self.try_load_by_filename(filename)

    def try_load_by_filename(self, filename):
        mem = self.mem

        if not fileutil.file_exists(filename):
            print('File: ', filename, "doesn't exist.")
            return False

        with open(filename, 'rb') as f:
            mem.t, mem.transitions = pickle.load(f)
            return True

    def save(self, action_index=-1, file_index=-1):
        """
        Save into a file.
        """
        if action_index == -1 or file_index == -1:
            filename = self.FINAL_SAVE_FILENAME
        else:
            filename = self._get_serialized_memory_filename(action_index, file_index)

        mem = self.mem

        with open(filename, 'wb') as f:
            pickle.dump([mem.t, mem.transitions], f)

    def _get_serialized_memory_filename(self, action_index, file_index):
        return self.MEMORY_FILENAME_TEMPLATE % (action_index, file_index)
