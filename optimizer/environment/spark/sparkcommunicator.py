import subprocess
import os
import time
from typing import List

from optimizer.environment.abstractcommunicator import AbstractCommunicator
from optimizer.environment.resetablecommunicator import ResetableCommunicator
from optimizer.environment.yarn.yarnmodel import FinishedApplication
from optimizer.util import processutil, jsonutil
from optimizer import hyperparameters


class SparkCommunicator(AbstractCommunicator, ResetableCommunicator):

    def __init__(self, rm_host: str, spark_history_server_host: str,
                 hadoop_home: str, spark_home: str, java_home: str):
        super().__init__(rm_host, spark_history_server_host, hadoop_home)
        self.SPARK_HOME = spark_home
        self.JAVA_HOME = java_home
        self.workload_runner = SparkWorkloadController()

    def is_done(self) -> bool:
        return False
        # return self.workload_runner.is_done()

    def close(self):
        return
        self.workload_runner.stop_workloads()
        restart_process = restart_yarn(os.getcwd(), self.HADOOP_HOME)
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

    @staticmethod
    def get():
        url = 'http://omnisky:8088/' + 'ws/v1/cluster/apps?states=FINISHED'
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
    QUEUES = hyperparameters.QUEUES['names']
    OPTIONS = [*map(lambda item: str(item), [1, 2, 3])]

    def __init__(self):
        self.processes: List[subprocess.Popen] = []

    def start_workloads(self, wd, spark_home, hadoop_home, java_home):
        ITEMS = [['bayes', 'queueA', '1'], ['bayes', 'queueA', '2'], ['bayes', 'queueA', '3'], ['fpgrowth', 'queueA', '1'], ['fpgrowth', 'queueB', '2'], ['fpgrowth', 'queueC', '3'], ['kmeans', 'queueC', '1'], ['kmeans', 'queueD', '2'], ['kmeans', 'queueA', '3'], ['lda', 'queueA', '1'], ['lda', 'queueD', '2'], ['lda', 'queueB', '3'], ['linear', 'queueC', '1'], ['linear', 'queueD', '2'], ['linear', 'queueB', '3'], ['SVM', 'queueC', '1'], ['SVM', 'queueB', '2'], ['SVM', 'queueD', '3'], ['als', 'queueA', '1'], ['als', 'queueB', '2'], ['als', 'queueB', '3']]
        for workload, queue, option in ITEMS:
            print('starting', workload, queue, option)
            p = start_workload_process(workload, queue, option, wd, spark_home, hadoop_home, java_home)
            self.processes.append(p)

    def stop_workloads(self):
        for p in self.processes:
            processutil.kill_process_and_wait(p)
        self.processes.clear()

    def is_done(self):
        return all([processutil.has_process_finished(p) for p in self.processes])


def start_workload_process(workload_type, queue, option, wd, spark_home, hadoop_home, java_home):
    c = ['%s/bin/start-spark-workload.sh' % wd, workload_type, spark_home, hadoop_home,  java_home, wd, queue, option]
    return subprocess.Popen(' '.join(c), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
