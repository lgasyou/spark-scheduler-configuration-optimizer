from optimizer.environment.spark import sparkmodel
from optimizer.environment.spark import finishedsparkmodel


class FinishedSparkApplicationAnalyzer(object):

    def __init__(self):
        self.sizes = [2**(i + 20) for i in range(12)]

    def analyze_and_save(self, application: sparkmodel.Application):
        app = finishedsparkmodel.CompletedSparkApplication(application.application_id)
        original_input_bytes = application.jobs[-1].stages[-1].input_bytes
        for job in application.jobs:
            for stage in job.stages:
                if not stage.tasks:
                    continue

                stage_name = stage.name
                input_bytes = stage.input_bytes
                sum_duration = sum([task.duration for task in stage.tasks])
                bytes_per_ms = input_bytes / sum_duration

                bytes_each_block = self.get_block_size(input_bytes / stage.num_tasks)

                if stage_name in app.computed_stage:
                    modeled_stage = app.computed_stage[stage_name]
                elif bytes_per_ms:
                    modeled_stage = app.computed_stage[stage_name] = []

                if bytes_per_ms:
                    modeled_stage.append(bytes_per_ms)
                    app.stages.append(finishedsparkmodel.ModelStage(stage_name,
                                                                    input_bytes / original_input_bytes,
                                                                    bytes_each_block))

        for k, v in app.computed_stage.items():
            app.computed_stage[k] = sum(v) / len(v)

        return app

    def get_block_size(self, bytes_each_task):
        if bytes_each_task < 2**20:
            return -1

        for size in self.sizes:
            if bytes_each_task < size:
                return size
