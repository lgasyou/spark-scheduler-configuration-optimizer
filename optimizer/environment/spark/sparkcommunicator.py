import subprocess
import os
import time
from typing import List

from optimizer.environment.abstractcommunicator import AbstractCommunicator
from optimizer.environment.resetablecommunicator import ResetableCommunicator
from optimizer.environment.yarn.yarnmodel import FinishedApplication
from optimizer.util import processutil, jsonutil


class SparkCommunicator(AbstractCommunicator, ResetableCommunicator):

    def __init__(self, rm_host: str, spark_history_server_host: str,
                 hadoop_home: str, spark_home: str, java_home: str):
        super().__init__(rm_host, spark_history_server_host, hadoop_home)
        self.SPARK_HOME = spark_home
        self.JAVA_HOME = java_home
        self.workload_runner = SparkWorkloadController()

    def is_done(self) -> bool:
        return False
        return self.workload_runner.is_done()

    def close(self):
        return
        self.workload_runner.stop_workloads()
        restart_process = restart_yarn(os.getcwd(), self.hadoop_home)
        restart_process.wait()

    def reset(self):
        return
        self.close()
        self.start_workload()
        time.sleep(5)

    def get_total_time_cost(self):
        url = self.RM_API_URL + 'ws/v1/cluster/apps?states=FINISHED'
        job_json = jsonutil.get_json(url)
        finished_jobs = build_finished_jobs_from_json(job_json)
        time_costs = [j.elapsed_time for j in finished_jobs]
        return time_costs, sum(time_costs)

    def start_workload(self):
        self.workload_runner.start_workloads(os.getcwd(), self.SPARK_HOME, self.HADOOP_HOME, self.JAVA_HOME)

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


def restart_yarn(wd, hadoop_home):
    cmd = "%s/bin/restart-yarn.sh %s" % (wd, hadoop_home)
    return subprocess.Popen(cmd, shell=True)


def build_finished_jobs_from_json(j: dict) -> List[FinishedApplication]:
    if j['apps'] is None:
        return []

    apps, jobs = j['apps']['app'], []
    for j in apps:
        elapsed_time = j['elapsedTime']
        jobs.append(FinishedApplication(elapsed_time))

    return jobs
