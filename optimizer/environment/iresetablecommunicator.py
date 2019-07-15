from typing import Tuple

import pandas as pd

from optimizer.environment.icommunicator import ICommunicator


class IResetableCommunicator(ICommunicator):

    def reset(self):
        pass

    def close(self):
        pass

    def get_total_time_cost(self) -> Tuple[pd.DataFrame, int]:
        pass
