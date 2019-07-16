from typing import Tuple

import pandas as pd

from optimizer.environment.communicator import Communicator


class ResetableCommunicator(Communicator):

    def reset(self):
        pass

    def close(self):
        pass

    def get_total_time_cost(self) -> Tuple[pd.DataFrame, int]:
        pass
