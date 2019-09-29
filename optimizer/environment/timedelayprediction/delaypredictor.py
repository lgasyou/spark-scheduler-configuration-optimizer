import logging
from typing import List

from optimizer.environment.stateobtaining import yarnmodel
from optimizer.environment.timedelayprediction import sparkmodel
from optimizer.environment.timedelayprediction.resourceallocationsimulator import ResourceAllocationSimulator
from optimizer.environment.timedelayprediction.singledelaypredictor import SingleDelayPredictor
from optimizer.environment.timedelayprediction.sparkapplicationbuilder import SparkApplicationBuilder


class DelayPredictor(object):

    DEFAULT_INPUT_BYTES = 1024**3
    DEFAULT_NUM_EXECUTORS = 5

    def __init__(self, spark_history_server_api_url: str):
        self.logger = logging.getLogger(__name__)
        self.spark_application_builder = SparkApplicationBuilder(spark_history_server_api_url)
        self.single_predictor = SingleDelayPredictor(spark_history_server_api_url)
        self.simulator = ResourceAllocationSimulator()

    def predict(self, resources: list, running_apps: List[yarnmodel.RunningApplication],
                waiting_apps: List[yarnmodel.WaitingApplication]):
        self.simulator.set_resources(resources)
        false_running_apps = self.predict_running_apps(running_apps)
        self.predict_false_running_apps(false_running_apps)
        self.predict_waiting_apps(waiting_apps)

    def predict_running_app(self, application_id: str, algorithm_type: str, start_time: int) -> tuple:
        app = self.spark_application_builder.build_partial_application(application_id)
        return self.single_predictor.predict(algorithm_type, app.input_bytes, app.executors, start_time)

    def predict_running_apps(self, running_apps):
        false_running_apps, real_running_apps = [], []
        for app in running_apps:
            app_id = app.application_id
            name = app.name
            start_time = app.started_time
            queue = app.converted_location
            delay, finish_time = self.predict_running_app(app_id, name, start_time)
            app.predicted_delay = delay
            if delay != -1:
                self.logger.info('%s, Delay: %f' % (app_id, delay))
                num_containers = app.running_containers
                self.simulator.release(finish_time, queue, num_containers)
            else:
                false_running_apps.append(app)
        return false_running_apps

    # Predict applications who are only allocated AM's container.
    def predict_false_running_apps(self, false_running_apps):
        false_running_apps.sort(key=lambda i: i.application_id)
        for app in false_running_apps:
            app_id = app.application_id
            name = app.name
            start_time = app.started_time
            queue = app.converted_location
            allocated = self.simulator.allocate(queue, self.DEFAULT_NUM_EXECUTORS)
            executors = self._build_executors(allocated)
            delay, finish_time = self.single_predictor.predict(name, self.DEFAULT_INPUT_BYTES, executors, start_time)
            app.predicted_delay = delay
            self.logger.info('%s, Delay: %f' % (app_id, delay))
            self.simulator.release(finish_time, queue, len(executors))

    def predict_waiting_apps(self, waiting_apps):
        waiting_apps.sort(key=lambda i: i.application_id)
        for app in waiting_apps:
            app_id = app.application_id
            name = app.name
            start_time = app.started_time
            queue = app.converted_location
            allocated = self.simulator.allocate(queue, self.DEFAULT_NUM_EXECUTORS)
            executors = self._build_executors(allocated)[1:]
            delay, finish_time = self.single_predictor.predict(name, self.DEFAULT_INPUT_BYTES, executors, start_time)
            app.predicted_delay = delay
            self.logger.info('%s, Delay: %f' % (app_id, delay))
            self.simulator.release(finish_time, queue, len(executors) + 1)

    @staticmethod
    def _build_executors(allocated_containers: list) -> List[sparkmodel.Executor]:
        executors = []
        for idx, step in enumerate(allocated_containers):
            time, num_containers = step
            executors.extend([sparkmodel.Executor(idx, True, time)] * num_containers)
        return executors
