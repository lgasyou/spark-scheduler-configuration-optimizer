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


class Action(object):
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


class YarnSchedulerCommunicator(object):
    """
    Manages communications with YARN cluster scheduler.
    After changing the capacity of queues, command `yarn rmadmin -refreshQueues` may be useful.
    """

    @staticmethod
    def get_action_set() -> Dict[int, Action]:
        """
        :return: Action dictionary defined in document.
        """
        return {
            1: Action(1, 5),
            2: Action(2, 4),
            3: Action(3, 3),
            4: Action(4, 2),
            5: Action(5, 1)
        }

    def act(self, action: Action) -> float:
        """
        Apply action and see how many rewards we can get.
        :return: Reward this step got.
        """
        self.save_conf()
        return 0

    def get_state(self) -> torch.Tensor:
        """
        Get raw state of YARN which has a variable length.
        """
        pass

    def get_state_trimmed(self) -> torch.Tensor:
        """
        Get state of YARN which is trimmed and the of length of it is determined,
        Which is the ϕ(s) function defined in document.
        """
        pass

    def save_conf(self) -> None:
        pass

    def is_done(self) -> bool:
        pass

    def close(self) -> None:
        pass
