import os
import threading
from typing import Optional

from optimizer.environment.clustercommunication.abstractcommunicator import AbstractCommunicator
from optimizer.environment.clustercommunication.ievaluationcommunicator import IEvaluationCommunicator
from optimizer.environment.workloadgenerating.workloadgenerator import WorkloadGenerator
from optimizer.util import yarnutil, sparkutil, processutil


class TrainingCommunicator(AbstractCommunicator, IEvaluationCommunicator):

    def __init__(self, rm_host: str, spark_history_server_host: str,
                 hadoop_home: str, spark_home: str, java_home: str):
        super().__init__(rm_host, spark_history_server_host, hadoop_home)
        self.SPARK_HOME = spark_home
        self.JAVA_HOME = java_home
        self.workload_generator = WorkloadGenerator()
        self.workload_starter: Optional[threading.Thread] = None

    def is_done(self) -> bool:
        return yarnutil.has_all_application_done(self.RM_API_URL) and \
               processutil.has_thread_finished(self.workload_starter)

    def close(self):
        self.logger.info('Restarting YARN...')
        restart_process = yarnutil.restart_yarn(os.getcwd(), self.HADOOP_HOME)
        restart_process.wait()
        self.logger.info('YARN restarted.')

    def reset(self):
        self.close()
        self.start_workloads()

    def generate_train_set(self) -> dict:
        return self.workload_generator.generate_randomly(batch_size=240)

    def start_workloads(self):
        workloads = self.generate_train_set()
        self.logger.info('Starting workloads...')
        self.workload_starter = sparkutil.async_start_workloads(workloads, self.SPARK_HOME,
                                                                self.HADOOP_HOME, self.JAVA_HOME)
        self.logger.info('Workloads started.')

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"
