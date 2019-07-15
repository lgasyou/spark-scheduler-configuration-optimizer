from typing import Dict, List

from optimizer.environment.spark import predictionsparkmodel, simulationmodel, sparkmodel
from optimizer.environment.spark.applicationexecutionsimulator import ApplicationExecutionSimulator
from optimizer.environment.spark.sparkapplicationbuilder import SparkApplicationBuilder


class SparkApplicationTimeDelayPredictor(object):

    def __init__(self, spark_history_server_api_url: str):
        self.spark_history_server_api_url = spark_history_server_api_url
        self.spark_application_builder = SparkApplicationBuilder(self.spark_history_server_api_url)
        self.simulator = ApplicationExecutionSimulator()

        self.models: Dict[str, predictionsparkmodel.Application] = {}
        self.task_id = 0

    def add_algorithm(self, algorithm_type: str, model: predictionsparkmodel.Application):
        self.models[algorithm_type] = model

    def predict(self, application_id: str, algorithm_type: str) -> int:
        app = self.spark_application_builder.build(application_id)
        return self._predict(algorithm_type, app.input_bytes, app.executors)

    def _predict(self, algorithm_type: str, input_bytes: int, executors: List[sparkmodel.Executor]) -> int:
        self.task_id = 0
        model = self.models[algorithm_type]

        tasks = self._build_tasks(input_bytes, model)
        containers = self._build_containers(executors)
        return self.simulator.simulate(containers, tasks)

    def _build_tasks(self, input_bytes: int, model: predictionsparkmodel.Application):
        tasks = []
        for stage in model.stages:
            stage_name = stage.name
            block_size = stage.block_size
            stage_input_bytes = input_bytes * stage.input_ratio
            num_tasks = int(stage_input_bytes / block_size)
            process_rate = model.average_action_process_rates[stage_name]

            for _ in range(num_tasks):
                tasks.append(simulationmodel.Task(self.task_id, block_size / process_rate))
                self.task_id += 1

            last_task_input_bytes = stage_input_bytes % block_size
            if last_task_input_bytes:
                tasks.append(simulationmodel.Task(self.task_id, last_task_input_bytes / process_rate))
                self.task_id += 1

        return tasks

    @staticmethod
    def _build_containers(executors: List[sparkmodel.Executor]):
        return [simulationmodel.Container(e.start_time, e.is_active) for e in executors]
