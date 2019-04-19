from typing import Tuple, Optional

import pandas as pd
import torch


class Communicator(object):
    """
    The interface of Communicator.
    """

    @staticmethod
    def get_action_set() -> dict:
        pass

    def get_state_tensor(self) -> Optional[torch.Tensor]:
        pass

    def act(self, action) -> float:
        pass

    def reset(self) -> None:
        pass

    def is_done(self) -> bool:
        pass

    def get_total_time_cost(self) -> Tuple[pd.DataFrame, int]:
        pass

    def close(self) -> None:
        pass
