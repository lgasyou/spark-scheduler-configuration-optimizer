import subprocess
import os
import time
from typing import Optional

from .abstractyarncommunicator import AbstractYarnCommunicator
from optimizer.util import processutil


class SparkCommunicator(AbstractYarnCommunicator):

    def __init__(self, api_url: str, hadoop_home: str, spark_home: str):
        super().__init__(api_url, hadoop_home)
        self.spark_home = spark_home
        self.workload_runner: Optional[subprocess.Popen] = None

    def is_done(self) -> bool:
        return processutil.has_process_finished(self.workload_runner)

    def close(self):
        if self.workload_runner is not None:
            processutil.kill_process_and_wait(self.workload_runner)
            self.workload_runner = None
            time.sleep(5)

    def start_workload(self):
        self.workload_runner = start_workload_process(os.getcwd(), self.spark_home)
        time.sleep(5)

    def get_scheduler_type(self) -> str:
        return "CapacityScheduler"


def start_workload_process(wd, spark_home):
    cmd = "%s/bin/start-spark-workload.sh %s" % (wd, spark_home)
    return subprocess.Popen(cmd, shell=True)
