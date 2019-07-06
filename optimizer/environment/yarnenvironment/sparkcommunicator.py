import subprocess
import os
import time
from typing import List

from .abstractyarncommunicator import AbstractYarnCommunicator
from .iresetablecommunicator import IResetableCommunicator
from .yarnmodel import FinishedJob
from optimizer.util import processutil, jsonutil


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
        restart_process = restart_yarn(os.getcwd())
        restart_process.wait()

    def reset(self):
        self.close()
        self.start_workload()
        time.sleep(5)

    def get_total_time_cost(self):
        url = self.api_url + 'ws/v1/cluster/apps?states=FINISHED'
        job_json = jsonutil.get_json(url)
        finished_jobs = build_finished_jobs_from_json(job_json)
        time_costs = [j.elapsed_time for j in finished_jobs]
        return time_costs, sum(time_costs)

    def start_workload(self):
        self.workload_runner.start_workloads(os.getcwd(), self.spark_home, self.hadoop_home, self.java_home)

    def get_scheduler_type(self) -> str:
        return "CapacityScheduler"


class SparkWorkloadController(object):

    WORKLOADS = ['bayes', 'fpgrowth', 'kmeans', 'lda', 'linear', 'SVM', 'als']

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


def restart_yarn(wd):
    cmd = "%s/bin/restart-yarn.sh" % wd
    return subprocess.Popen(cmd, shell=True)


def build_finished_jobs_from_json(j: dict) -> List[FinishedJob]:
    if j['apps'] is None:
        return []

    apps, jobs = j['apps']['app'], []
    for j in apps:
        elapsed_time = j['elapsedTime']
        jobs.append(FinishedJob(elapsed_time))

    return jobs
