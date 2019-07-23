import os

from optimizer.environment.clustercommunication.abstractcommunicator import AbstractCommunicator
from optimizer.environment.clustercommunication.ievaluationcommunicator import IEvaluationCommunicator
from optimizer.environment.workloadgenerating.workloadgenerator import WorkloadGenerator
from optimizer.util import yarnutil, sparkutil


class EvaluationCommunicator(AbstractCommunicator, IEvaluationCommunicator):

    def __init__(self, rm_host: str, spark_history_server_host: str,
                 hadoop_home: str, spark_home: str, java_home: str):
        super().__init__(rm_host, spark_history_server_host, hadoop_home)
        self.SPARK_HOME = spark_home
        self.JAVA_HOME = java_home
        # self.workload_generator = WorkloadGenerator()
        # self.WORKLOADS = self.workload_generator.load_evaluation_workloads()
        self.WORKLOADS = {'workloads': [
            {'name': 'bayes', 'interval': 5, 'queue': 'queueA', 'dataSize': '3'},
            {'name': 'svm', 'interval': 3, 'queue': 'queueA', 'dataSize': '3'},
            {'name': 'lda', 'interval': 2, 'queue': 'queueA', 'dataSize': '2'},
            {'name': 'FPGrowth', 'interval': 25, 'queue': 'queueB', 'dataSize': '4'},
            {'name': 'bayes', 'interval': 9, 'queue': 'queueB', 'dataSize': '3'},
            {'name': 'svm', 'interval': 2, 'queue': 'queueB', 'dataSize': '3'},
            {'name': 'lda', 'interval': 7, 'queue': 'queueA', 'dataSize': '2'},
            {'name': 'FPGrowth', 'interval': 1, 'queue': 'queueA', 'dataSize': '4'},
            {'name': 'svm', 'interval': 3, 'queue': 'queueA', 'dataSize': '3'},
            {'name': 'lda', 'interval': 1, 'queue': 'queueA', 'dataSize': '1'},
            {'name': 'lda', 'interval': 7, 'queue': 'queueB', 'dataSize': '3'},
            {'name': 'kmeans', 'interval': 9, 'queue': 'queueA', 'dataSize': '2'},
            {'name': 'FPGrowth', 'interval': 13, 'queue': 'queueB', 'dataSize': '6'},
            {'name': 'lda', 'interval': 9, 'queue': 'queueB', 'dataSize': '2'},
            {'name': 'svm', 'interval': 2, 'queue': 'queueA', 'dataSize': '4'},
            {'name': 'bayes', 'interval': 3, 'queue': 'queueA', 'dataSize': '3'},
            {'name': 'FPGrowth', 'interval': 0, 'queue': 'queueA', 'dataSize': '3'},
            {'name': 'kmeans', 'interval': 20, 'queue': 'queueA', 'dataSize': '3'},
            {'name': 'FPGrowth', 'interval': 10, 'queue': 'queueA', 'dataSize': '3'},
            {'name': 'bayes', 'interval': 7, 'queue': 'queueA', 'dataSize': '2'},
            {'name': 'lda', 'interval': 6, 'queue': 'queueB', 'dataSize': '3'},
            {'name': 'svm', 'interval': 98, 'queue': 'queueA', 'dataSize': '3'},
            {'name': 'kmeans', 'interval': 10, 'queue': 'queueA', 'dataSize': '3'},
            {'name': 'bayes', 'interval': 17, 'queue': 'queueB', 'dataSize': '2'},
            {'name': 'linear', 'interval': 31, 'queue': 'queueB', 'dataSize': '5'},
            {'name': 'lda', 'interval': 3, 'queue': 'queueA', 'dataSize': '2'},
            {'name': 'FPGrowth', 'interval': 2, 'queue': 'queueB', 'dataSize': '1'},
            {'name': 'FPGrowth', 'interval': 7, 'queue': 'queueA', 'dataSize': '2'},
            {'name': 'linear', 'interval': 1, 'queue': 'queueB', 'dataSize': '1'},
            {'name': 'kmeans', 'interval': 1, 'queue': 'queueA', 'dataSize': '1'},
            {'name': 'svm', 'interval': 3, 'queue': 'queueA', 'dataSize': '6'},
            {'name': 'lda', 'interval': 27, 'queue': 'queueA', 'dataSize': '3'},
            {'name': 'linear', 'interval': 125, 'queue': 'queueB', 'dataSize': '5'},
            {'name': 'kmeans', 'interval': 1, 'queue': 'queueB', 'dataSize': '1'},
            {'name': 'lda', 'interval': 4, 'queue': 'queueB', 'dataSize': '1'},
            {'name': 'lda', 'interval': 39, 'queue': 'queueB', 'dataSize': '4'}
        ]}

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
        finished_jobs = self.state_builder.parse_and_build_finished_apps()
        time_costs = [j.elapsed_time for j in finished_jobs]
        return time_costs, sum(time_costs)

    def start_workload(self):
        sparkutil.async_start_workloads(self.WORKLOADS, self.SPARK_HOME, self.HADOOP_HOME, self.JAVA_HOME)

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"
