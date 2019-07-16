from typing import List, Optional

from optimizer.environment.spark import sparkmodel, predictionsparkmodel


class CompletedSparkApplicationAnalyzer(object):

    def __init__(self):
        self.ERROR_RATE = 0.2
        self.BLOCK_SIZES = [2 ** (20 + i) for i in range(6, 12)]  # From 64MB to 4GB
        self.SIZE_ERRORS = [self.ERROR_RATE * size for size in self.BLOCK_SIZES]
        self.cur_app: Optional[predictionsparkmodel.Application] = None

    def analyze(self, application: sparkmodel.Application) -> predictionsparkmodel.Application:
        self.cur_app = predictionsparkmodel.Application(application.application_id)
        original_input_bytes = application.input_bytes
        for job in application.jobs:
            for stage in job.stages:
                input_bytes = stage.input_bytes
                if not stage.tasks or not input_bytes:
                    continue

                stage_name = stage.name
                input_ratio = input_bytes / original_input_bytes
                block_size = self._get_block_size(input_bytes, stage.num_tasks)
                model_stage = predictionsparkmodel.Stage(stage_name, input_ratio, block_size)
                self.cur_app.stages.append(model_stage)

                average_process_rate = self._get_average_process_rate(stage.tasks, block_size)
                self._add_action_process_rate(stage_name, average_process_rate)

        self._calculate_actions_average_process_rate()
        return self.cur_app

    def _add_action_process_rate(self, stage_name: str, process_rate: float):
        if stage_name in self.cur_app.average_action_process_rates:
            modeled_stage = self.cur_app.average_action_process_rates[stage_name]
        else:
            modeled_stage = self.cur_app.average_action_process_rates[stage_name] = []
        modeled_stage.append(process_rate)

    @staticmethod
    def _get_average_process_rate(tasks: List[sparkmodel.Task], block_size: int):
        precision = 0.02 * block_size   # Only calculate tasks with full block-size input.
        process_rates = [t.input_bytes / t.duration for t in tasks if abs(t.input_bytes - block_size) < precision]
        if not process_rates:   # If we don't have any task with block-size input, use all of them.
            process_rates = [t.input_bytes / t.duration for t in tasks]
        average_process_rate = sum(process_rates) / len(process_rates)
        return average_process_rate

    def _calculate_actions_average_process_rate(self):
        for k, v in self.cur_app.average_action_process_rates.items():
            self.cur_app.average_action_process_rates[k] = sum(v) / len(v)

    def _get_block_size(self, input_bytes: int, num_tasks: int):
        return input_bytes if num_tasks == 1 else self._block_size(input_bytes / num_tasks)

    def _block_size(self, bytes_each_task):
        for size, precision in zip(self.BLOCK_SIZES, self.SIZE_ERRORS):
            if abs(bytes_each_task - size) < precision:
                return size

        return bytes_each_task
