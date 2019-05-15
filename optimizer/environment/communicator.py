from typing import Tuple

import pandas as pd
import torch


class ICommunicator(object):
    """
    The interface of Communicator.
    """

    @staticmethod
    def get_action_set() -> dict:
        pass

    def get_state_tensor(self) -> torch.Tensor:
        pass

    def act(self, action) -> float:
        pass

    def is_done(self) -> bool:
        pass


class IResetableCommunicator(ICommunicator):

    def reset(self):
        pass

    def close(self):
        pass

    def get_total_time_cost(self) -> Tuple[pd.DataFrame, int]:
        pass
