from typing import List

from optimizer.environment.spark.jobfinishtimepredictor import JobFinishTimePredictor
from optimizer.environment.spark import sparkmodel, calculationmodel
from optimizer.environment.spark.sparkapplicationbuilder import SparkApplicationBuilder


class SparkApplicationTimeDelayPredictor(object):

    def __init__(self, spark_history_server_api_url: str):
        self.spark_history_server_api_url = spark_history_server_api_url
        self.spark_application_builder = SparkApplicationBuilder(self.spark_history_server_api_url)

    def predict(self, application_id: str):
        application = self.spark_application_builder.build(application_id)
        return self._predict(application)

    # TODO: 从application中获取输入的数据大小，通过其他途径获得输入的算法类型
    # TODO: 从输入的以往数据中获取当前的Task数目
    def _predict(self, application: sparkmodel.Application):
        time_delay_predictor = JobFinishTimePredictor()
        containers = [calculationmodel.Container(e.start_time, e.is_active) for e in application.executors]
        tasks = self._build_tasks(application)
        finish_time = time_delay_predictor.simulate(containers, tasks)
        time_delay = finish_time - application.start_time
        return time_delay, finish_time

    @staticmethod
    def _build_tasks(application: sparkmodel.Application) -> List[calculationmodel.Task]:
        tasks = []
        for job in application.jobs:
            for stage in job.stages:
                for task in stage.tasks:
                    # TODO: Differences of task execution time
                    tasks.append(calculationmodel.Task(task.task_id, 26642))

        return tasks
