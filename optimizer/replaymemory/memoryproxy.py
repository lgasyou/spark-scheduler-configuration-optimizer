from typing import Optional

from .memory import ReplayMemory


class ReplayMemoryProxy(object):
    """
    Is the proxy of Replay Memory.
    To solve a problem which is caused when StateInvalidException raises.
    """

    def __init__(self, args, capacity):
        self._mem = ReplayMemory(args, capacity)
        self._cache: Optional[list] = None

    def append(self, state, action, reward, terminal):
        self._flush_cache()
        self._cache = [state, action, reward, terminal]

    def terminate(self):
        self._set_terminal_flag()
        self._flush_cache()

    def sample(self, batch_size):
        return self._mem.sample(batch_size)

    def update_priorities(self, idxs, priorities):
        return self._mem.update_priorities(idxs, priorities)

    @property
    def memory(self):
        return self._mem

    def _flush_cache(self):
        if self._cache is not None:
            state, action, reward, terminal = self._cache
            self._mem.append(state, action, reward, terminal)
            self._cache = None

    def _set_terminal_flag(self):
        if self._cache is not None:
            self._cache[3] = True

    # Set up internal state for iterator
    def __iter__(self):
        return self._mem.__iter__()

    # Return valid states for validation
    def __next__(self):
        return self._mem.__next__()
