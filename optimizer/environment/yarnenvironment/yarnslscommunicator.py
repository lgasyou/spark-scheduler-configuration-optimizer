import os
import signal
import subprocess
import time
from typing import Optional

import pandas as pd
import psutil

from .abstractyarncommunicator import AbstractYarnCommunicator
from .iresetablecommunicator import IResetableCommunicator


class YarnSlsCommunicator(AbstractYarnCommunicator, IResetableCommunicator):

    def __init__(self, api_url: str, hadoop_home: str, sls_jobs_dataset: str = None):
        super().__init__(api_url, hadoop_home)
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
        self.sls_runner = start_sls_process(wd, self.hadoop_home, sls_jobs_json)

        # Wait until web server starts.
        time.sleep(10)

    def close(self) -> None:
        """
        Kills SLS process.
        """
        if self.sls_runner is not None:
            kill_process_family(self.sls_runner.pid)
            self.sls_runner.wait()
            self.sls_runner = None
            time.sleep(5)

    def is_done(self) -> bool:
        return self.sls_runner is None or self.sls_runner.poll() is not None

    def get_scheduler_type(self) -> str:
        return "CapacityScheduler"

    def get_total_time_cost(self):
        data = pd.read_csv('./results/logs/jobruntime.csv')
        end_time = data['simulate_end_time']
        start_time = data['simulate_start_time']
        time_costs = end_time - start_time
        sum_time_cost = time_costs.sum()
        return time_costs, sum_time_cost


def kill_process_family(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        for process in children:
            process.send_signal(sig)
        parent.send_signal(sig)
    except psutil.NoSuchProcess:
        return


def start_sls_process(wd: str, hadoop_home: str, sls_jobs_json: str):
    cmd = "%s/bin/start-sls.sh %s %s %s" % (wd, hadoop_home, wd, sls_jobs_json)
    return subprocess.Popen(cmd, shell=True)
