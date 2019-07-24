import os
import threading
from typing import Optional

from optimizer.environment.clustercommunication.abstractcommunicator import AbstractCommunicator
from optimizer.environment.clustercommunication.ievaluationcommunicator import IEvaluationCommunicator
from optimizer.environment.workloadgenerating.workloadgenerator import WorkloadGenerator
from optimizer.util import yarnutil, sparkutil, processutil


class EvaluationCommunicator(AbstractCommunicator, IEvaluationCommunicator):

    def __init__(self, rm_host: str, spark_history_server_host: str,
                 hadoop_home: str, spark_home: str, java_home: str):
        super().__init__(rm_host, spark_history_server_host, hadoop_home)
        self.SPARK_HOME = spark_home
        self.JAVA_HOME = java_home
        self.workload_generator = WorkloadGenerator()
        self.WORKLOADS = self.workload_generator.load_evaluation_workloads(100)
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

    def get_total_time_cost(self):
        finished_jobs = self.state_builder.parse_and_build_finished_apps()
        time_costs = [j.elapsed_time for j in finished_jobs]
        return time_costs, sum(time_costs)

    def start_workloads(self):
        self.workload_starter = sparkutil.async_start_workloads(self.WORKLOADS, self.SPARK_HOME,
                                                                self.HADOOP_HOME, self.JAVA_HOME)

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"
