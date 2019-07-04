import subprocess
import os
import time
from typing import List

from .abstractyarncommunicator import AbstractYarnCommunicator
from .iresetablecommunicator import IResetableCommunicator
from optimizer.util import processutil


class SparkCommunicator(AbstractYarnCommunicator, IResetableCommunicator):

    def __init__(self, api_url: str, hadoop_home: str, spark_home: str, java_home: str):
        super().__init__(api_url, hadoop_home)
        self.spark_home = spark_home
        self.java_home = java_home
        self.workload_runner = SparkWorkloadController()

    def is_done(self) -> bool:
        return self.workload_runner.is_done()

    def close(self):
        self.workload_runner.stop_workloads()
        time.sleep(5)

    def reset(self):
        self.close()
        self.start_workload()
        time.sleep(5)

    def start_workload(self):
        self.workload_runner.start_workloads(os.getcwd(), self.spark_home, self.hadoop_home, self.java_home)

    def get_scheduler_type(self) -> str:
        return "CapacityScheduler"


class SparkWorkloadController(object):

    WORKLOADS = ['bayes', 'fpgrowth', 'kmeans', 'lda', 'linear', 'SVM']

    def __init__(self):
        self.processes: List[subprocess.Popen] = []

    def start_workloads(self, wd, spark_home, hadoop_home, java_home):
        for workload in self.WORKLOADS:
            p = start_workload_process(workload, wd, spark_home, hadoop_home, java_home)
            self.processes.append(p)

    def stop_workloads(self):
        for p in self.processes:
            processutil.kill_process_and_wait(p)
        self.processes.clear()

    def is_done(self):
        return all([processutil.has_process_finished(p) for p in self.processes])


def start_workload_process(workload_type, wd, spark_home, hadoop_home, java_home):
    cmd = "%s/bin/start-spark-workload.sh %s %s %s %s %s" % (wd, workload_type, spark_home, hadoop_home, java_home, wd)
    return subprocess.Popen(cmd, shell=True)
