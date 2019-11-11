import threading
from typing import Tuple

import pandas as pd

from optimizer.environment.clustercommunication.icommunicator import ICommunicator


class IEvaluationCommunicator(ICommunicator):

    def reset(self):
        pass

    def close(self):
        pass

    def get_finished_job_number(self) -> Tuple[pd.DataFrame, int]:
        pass

    def start_workloads(self) -> threading.Thread:
        pass
