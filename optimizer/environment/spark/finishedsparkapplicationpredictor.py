from optimizer.environment.spark import finishedsparkmodel, calculationmodel, sparkmodel
from optimizer.environment.spark.jobfinishtimepredictor import JobFinishTimePredictor


class FinishedSparkApplicationPredictor(object):

    def __init__(self, ):
        self.finish_time_predictor = JobFinishTimePredictor()
        self.algorithms = {}

    def add_algorithm(self, algorithm_type: str, model: finishedsparkmodel.CompletedSparkApplication):
        self.algorithms[algorithm_type] = model

    def predict(self, algorithm_type: str, input_bytes: int, application: sparkmodel.Application):
        algor = self.algorithms[algorithm_type]
        task_id = 0
        tasks = []
        for stage in algor.stages:
            stage_name = stage.name
            input_rate = stage.input_size_rate
            stage_input_bytes = input_bytes * input_rate
            num_tasks = int(stage_input_bytes / stage.bytes_each_task) + 1
            last_task_input_bytes = stage_input_bytes % stage.bytes_each_task
            process_rate = algor.computed_stage[stage_name]

            for _ in range(num_tasks - 1):
                tasks.append(calculationmodel.Task(task_id, stage.bytes_each_task / process_rate))
                task_id += 1
            tasks.append(calculationmodel.Task(task_id, last_task_input_bytes / process_rate))
            task_id += 1

        containers = self.container_builder(application)
        return self.finish_time_predictor.simulate(containers, tasks) + 10000

    def container_builder(self, application):
        return [calculationmodel.Container(e.start_time, e.is_active) for e in application.executors]
