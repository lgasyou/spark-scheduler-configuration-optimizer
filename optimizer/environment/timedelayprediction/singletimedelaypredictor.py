from typing import List

from optimizer.environment.timedelayprediction import predictionsparkmodel, simulationmodel, sparkmodel
from optimizer.environment.timedelayprediction.algorithmmodelbuilder import AlgorithmModelBuilder
from optimizer.environment.timedelayprediction.applicationexecutionsimulator import ApplicationExecutionSimulator


class SingleTimeDelayPredictor(object):

    def __init__(self, spark_application_builder):
        self.simulator = ApplicationExecutionSimulator()
        algorithm_model_builder = AlgorithmModelBuilder(spark_application_builder)
        self.models = algorithm_model_builder.get_model()
        self.task_id = 0

    def add_algorithm(self, algorithm_type: str, model: predictionsparkmodel.Application):
        self.models[algorithm_type] = model

    # Returns time delay and finish time.
    def predict(self, algorithm_type: str, input_bytes: int,
                executors: List[sparkmodel.Executor], start_time) -> tuple:
        self.task_id = 0
        model = self.models[algorithm_type]

        tasks = self._build_tasks(input_bytes, model)
        containers = self._build_containers(executors)
        finish_time = self.simulator.simulate(containers, tasks)
        time_delay = (finish_time - start_time) / 1000 if finish_time > 0 else -1
        return time_delay, finish_time

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
