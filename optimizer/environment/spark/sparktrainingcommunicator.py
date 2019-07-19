import os
import threading

from optimizer.environment.abstractcommunicator import AbstractCommunicator
from optimizer.environment.spark.sparkworkloadrandomgenerator import SparkWorkloadRandomGenerator
from optimizer.util import processutil, jsonutil, sparkutil


class SparkTrainingCommunicator(AbstractCommunicator):

    def __init__(self, rm_host: str, spark_history_server_host: str,
                 hadoop_home: str, spark_home: str, java_home: str):
        super().__init__(rm_host, spark_history_server_host, hadoop_home)
        self.SPARK_HOME = spark_home
        self.JAVA_HOME = java_home
        self.workload_generator = SparkWorkloadRandomGenerator()

    def is_done(self) -> bool:
        url = self.RM_API_URL + 'ws/v1/cluster/apps?states=NEW,NEW_SAVING,SUBMITTED,ACCEPTED,RUNNING'
        app_json = jsonutil.get_json(url)
        all_app_finished = app_json['apps'] is None
        return all_app_finished

    def close(self):
        self.logger.info('Restarting YARN...')
        restart_process = restart_yarn(os.getcwd(), self.HADOOP_HOME)
        restart_process.wait()
        self.logger.info('YARN restarted.')

    def reset(self, workloads):
        self.close()
        self.start_workloads(workloads)

    def generate_pre_train_set(self) -> dict:
        return self.workload_generator.generate()

    def start_workloads(self, workloads):
        args = (workloads, self.SPARK_HOME, self.HADOOP_HOME, self.JAVA_HOME)
        start_thread = threading.Thread(target=sparkutil.spark_submit, args=args)
        start_thread.start()
        start_thread.join(30)

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"


def restart_yarn(wd, hadoop_home):
    cmd = ["%s/bin/restart-yarn.sh" % wd, hadoop_home]
    return processutil.start_process(cmd)