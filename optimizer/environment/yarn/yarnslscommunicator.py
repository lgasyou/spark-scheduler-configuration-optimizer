import os
import subprocess
import time
from typing import Optional

import pandas as pd

from optimizer.environment.abstractcommunicator import AbstractCommunicator
from optimizer.environment.resetablecommunicator import ResetableCommunicator
from optimizer.util import processutil


class YarnSlsCommunicator(AbstractCommunicator, ResetableCommunicator):

    def __init__(self, rm_host: str, spark_history_server_host: str,
                 hadoop_home: str, sls_jobs_dataset: str = ''):
        super().__init__(rm_host, spark_history_server_host, hadoop_home)
        self.current_dataset = sls_jobs_dataset
        self.sls_runner: Optional[subprocess.Popen] = None

    def set_dataset(self, filename: str):
        self.current_dataset = filename

    def reset(self) -> None:
        """
        Resets the environment in order to run this program again.
        """
        self.close()

        wd = os.getcwd()
        sls_jobs_json = './' + self.current_dataset
        self.sls_runner = start_sls_process(wd, self.HADOOP_HOME, sls_jobs_json)

        # Wait until web server starts.
        time.sleep(10)

    def close(self) -> None:
        """
        Kills SLS process.
        """
        if self.sls_runner is not None:
            processutil.kill_process_and_wait(self.sls_runner)
            self.sls_runner = None
            time.sleep(5)

    def is_done(self) -> bool:
        return processutil.has_process_finished(self.sls_runner)

    def get_scheduler_type(self) -> str:
        return "CapacityScheduler"

    def get_total_time_cost(self):
        data = pd.read_csv('./results/logs/jobruntime.csv')
        end_time = data['simulate_end_time']
        start_time = data['simulate_start_time']
        time_costs = end_time - start_time
        sum_time_cost = time_costs.sum()
        return time_costs, sum_time_cost


def start_sls_process(wd: str, hadoop_home: str, sls_jobs_json: str):
    cmd = "%s/bin/start-sls.sh %s %s %s" % (wd, hadoop_home, wd, sls_jobs_json)
    return subprocess.Popen(cmd, shell=True)
