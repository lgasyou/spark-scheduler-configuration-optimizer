from typing import List

from optimizer.environment.spark.applicationexecutionsimulator import ApplicationExecutionSimulator
from optimizer.environment.spark import sparkmodel, simulationmodel
from optimizer.environment.spark.sparkapplicationbuilder import SparkApplicationBuilder


@DeprecationWarning
class DynamicSparkApplicationTimeDelayPredictor(object):

    def __init__(self, spark_history_server_api_url: str):
        self.spark_history_server_api_url = spark_history_server_api_url
        self.spark_application_builder = SparkApplicationBuilder(self.spark_history_server_api_url)

    def predict(self, application_id: str):
        application = self.spark_application_builder.build(application_id)
        return self._predict(application)

    def _predict(self, application: sparkmodel.Application):
        time_delay_predictor = ApplicationExecutionSimulator()
        containers = [simulationmodel.Container(e.start_time, e.is_active) for e in application.executors]
        tasks = self._build_tasks(application)
        finish_time = time_delay_predictor.simulate(containers, tasks)
        time_delay = finish_time - application.start_time
        return time_delay, finish_time

    @staticmethod
    def _build_tasks(application: sparkmodel.Application) -> List[simulationmodel.Task]:
        tasks = []
        for job in application.jobs:
            for stage in job.stages:
                for task in stage.tasks:
                    # TODO: Differences of task execution time
                    tasks.append(simulationmodel.Task(task.task_id, 26642))

        return tasks
