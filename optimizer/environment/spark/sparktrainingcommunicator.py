import os
import time

from optimizer.environment.abstractcommunicator import AbstractCommunicator
from optimizer.environment.spark.sparkcommunicator import SparkWorkloadController
from optimizer.util import processutil, jsonutil


class SparkTrainingCommunicator(AbstractCommunicator):

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
        self.logger.info('{}{}'.format(all_app_finished, self.workload_runner.is_done()))
        return done

    def close(self):
        self.workload_runner.stop_workloads()
        self.logger.info('Restarting YARN...')
        restart_process = restart_yarn(os.getcwd(), self.HADOOP_HOME)
        restart_process.wait()
        self.logger.info('YARN restarted.')

    def reset(self, workloads):
        self.close()
        self.start_workloads(workloads)
        time.sleep(5)

    def generate_pre_train_set(self):
        pass

    def start_workloads(self, workloads):
        self.workload_runner.start_workloads(os.getcwd(), self.SPARK_HOME, self.HADOOP_HOME, self.JAVA_HOME)

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"


def restart_yarn(wd, hadoop_home):
    cmd = ["%s/bin/restart-yarn.sh" % wd, hadoop_home]
    return processutil.start_process(cmd)
