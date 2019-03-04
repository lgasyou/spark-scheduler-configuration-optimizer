from typing import Dict

import torch


class Job(object):
    def __init__(self):
        self.submit_time = int()
        self.priority = int()
        self.task = []


class Resource(object):
    def __init__(self):
        self.plat = ''
        self.cpu = ''
        self.mem = ''


class Constraint(object):
    def __init__(self):
        self.queue = []
        self.job = []


class State(object):
    def __init__(self, raw_state: torch.Tensor):
        self.job = Job()
        self.resource = Resource()
        self.constraint = Constraint()


class YarnAction(object):
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


class YarnSchedulerCommunicator(object):
    """
    Manages communications with YARN cluster scheduler.
    """

    @staticmethod
    def get_action_set() -> Dict[int, YarnAction]:
        return {
            1: YarnAction(1, 5),
            2: YarnAction(2, 4),
            3: YarnAction(3, 3),
            4: YarnAction(4, 2),
            5: YarnAction(5, 1)
        }

    def act(self, action: int) -> float:
        """
        :return: reward this step got.
        """
        pass

    def get_state(self) -> torch.Tensor:
        pass

    def save_conf(self) -> None:
        pass

    def is_done(self) -> bool:
        pass

    def close(self) -> None:
        pass
