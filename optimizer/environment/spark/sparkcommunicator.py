import logging
import os
import subprocess
import time
from typing import List

from optimizer import hyperparameters
from optimizer.environment.abstractcommunicator import AbstractCommunicator
from optimizer.environment.evaluationcommunicator import EvaluationCommunicator
from optimizer.environment.yarn.yarnmodel import FinishedApplication
from optimizer.util import processutil, jsonutil


class SparkCommunicator(AbstractCommunicator, EvaluationCommunicator):

    def __init__(self, rm_host: str, spark_history_server_host: str,
                 hadoop_home: str, spark_home: str, java_home: str):
        super().__init__(rm_host, spark_history_server_host, hadoop_home)
        self.SPARK_HOME = spark_home
        self.JAVA_HOME = java_home
        self.workload_runner = SparkWorkloadController()

    def is_done(self) -> bool:
        url = self.RM_API_URL + 'ws/v1/cluster/apps?states=NEW,NEW_SAVING,SUBMITTED,ACCEPTED,RUNNING'
        app_json = jsonutil.get_json(url)
        all_app_finished = app_json['apps'] is None
        done = all_app_finished and self.workload_runner.is_done()
        return done

    def close(self):
        self.workload_runner.stop_workloads()
        self.logger.info('Restarting YARN...')
        restart_process = restart_yarn(os.getcwd(), self.HADOOP_HOME)
        restart_process.wait()
        self.logger.info('YARN restarted.')

    def reset(self):
        self.close()
        self.start_workload()

    def get_total_time_cost(self):
        url = self.RM_API_URL + 'ws/v1/cluster/apps?states=FINISHED'
        job_json = jsonutil.get_json(url)
        finished_jobs = build_finished_jobs_from_json(job_json)
        time_costs = [j.elapsed_time for j in finished_jobs]
        return time_costs, sum(time_costs)

    def start_workload(self):
        self.workload_runner.start_workloads(os.getcwd(), self.SPARK_HOME, self.HADOOP_HOME, self.JAVA_HOME)

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"


class SparkWorkloadController(object):

    WORKLOADS = ['bayes', 'fpgrowth', 'kmeans', 'lda', 'linear', 'SVM']
    QUEUES = hyperparameters.QUEUES['names']
    OPTIONS = [*map(lambda item: str(item), [1, 2, 3])]

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processes: List[subprocess.Popen] = []

    def start_workloads(self, wd, spark_home, hadoop_home, java_home):
        ITEMS = [['bayes', 'queueA', '1'], ['bayes', 'queueB', '2'], ['bayes', 'queueA', '3'], ['fpgrowth', 'queueA', '1'], ['fpgrowth', 'queueB', '2'], ['fpgrowth', 'queueA', '3'], ['kmeans', 'queueB', '1'], ['kmeans', 'queueA', '2'], ['kmeans', 'queueA', '3'], ['lda', 'queueA', '1'], ['lda', 'queueB', '2'], ['lda', 'queueB', '3'], ['linear', 'queueA', '1'], ['linear', 'queueB', '2'], ['linear', 'queueB', '3'], ['SVM', 'queueA', '1'], ['SVM', 'queueB', '2'], ['SVM', 'queueB', '3']]
        for workload, queue, option in ITEMS:
            self.logger.info('Starting {} on {} with option {}...'.format(workload, queue, option))
            p = start_workload_process(workload, queue, option, wd, spark_home, hadoop_home, java_home)
            self.processes.append(p)
            time.sleep(1)

    def stop_workloads(self):
        for p in self.processes:
            processutil.kill_process_and_wait(p)
        self.processes.clear()

    def is_done(self):
        return all([processutil.has_process_finished(p) for p in self.processes])


def start_workload_process(workload_type, queue, option, wd, spark_home, hadoop_home, java_home):
    c = ['%s/bin/start-spark-workloadsubmission.sh' % wd, workload_type, spark_home, hadoop_home,  java_home, wd, queue, option]
    return processutil.start_process(c)


def restart_yarn(wd, hadoop_home):
    cmd = ["%s/bin/restart-yarn.sh" % wd, hadoop_home]
    return processutil.start_process(cmd)


def build_finished_jobs_from_json(j: dict) -> List[FinishedApplication]:
    if j['apps'] is None:
        return []

    apps, jobs = j['apps']['app'], []
    for j in apps:
        elapsed_time = j['elapsedTime']
        jobs.append(FinishedApplication(elapsed_time))

    return jobs
