import os
from typing import List

from optimizer.environment.abstractcommunicator import AbstractCommunicator
from optimizer.environment.evaluationcommunicator import EvaluationCommunicator
from optimizer.environment.yarn.yarnmodel import FinishedApplication
from optimizer.util import jsonutil, yarnutil, sparkutil


class SparkEvaluationCommunicator(AbstractCommunicator, EvaluationCommunicator):

    def __init__(self, rm_host: str, spark_history_server_host: str,
                 hadoop_home: str, spark_home: str, java_home: str):
        super().__init__(rm_host, spark_history_server_host, hadoop_home)
        self.SPARK_HOME = spark_home
        self.JAVA_HOME = java_home

        ITEMS = [['linear', 'queueB', '5'], ['bayes', 'queueA', '2'], ['linear', 'queueB', '8'],
                 ['kmeans', 'queueA', '8'], ['lda', 'queueA', '2'], ['FPGrowth', 'queueB', '5'],
                 ['lda', 'queueB', '5'], ['linear', 'queueA', '2'], ['lda', 'queueB', '8'],
                 ['FPGrowth', 'queueA', '2'], ['kmeans', 'queueA', '5'], ['svm', 'queueA', '2'],
                 ['FPGrowth', 'queueA', '8'], ['kmeans', 'queueB', '2'], ['svm', 'queueB', '5'],
                 ['bayes', 'queueB', '5'], ['bayes', 'queueA', '8']]
        self.WORKLOADS = {
            'workloads': [
                {
                    "name": name,
                    "interval": 5,
                    "queue": queue,
                    "dataSize": data_size
                } for name, queue, data_size in ITEMS
            ]
        }

    def is_done(self) -> bool:
        return yarnutil.has_all_application_done(self.RM_API_URL)

    def close(self):
        self.logger.info('Restarting YARN...')
        restart_process = yarnutil.restart_yarn(os.getcwd(), self.HADOOP_HOME)
        restart_process.wait()
        self.logger.info('YARN restarted.')

    def reset(self):
        self.close()
        self.start_workload()

    def get_total_time_cost(self):
        url = self.RM_API_URL + 'ws/v1/cluster/apps?states=FINISHED'
        job_json = jsonutil.get_json(url)
        finished_jobs = self.build_finished_jobs_from_json(job_json)
        time_costs = [j.elapsed_time for j in finished_jobs]
        return time_costs, sum(time_costs)

    @staticmethod
    def build_finished_jobs_from_json(j: dict) -> List[FinishedApplication]:
        if j['apps'] is None:
            return []

        apps, jobs = j['apps']['app'], []
        for j in apps:
            elapsed_time = j['elapsedTime']
            jobs.append(FinishedApplication(elapsed_time))

        return jobs

    def start_workload(self):
        self.logger.info(self.WORKLOADS)
        sparkutil.async_start_workloads(self.WORKLOADS, self.SPARK_HOME, self.HADOOP_HOME, self.JAVA_HOME)

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"
