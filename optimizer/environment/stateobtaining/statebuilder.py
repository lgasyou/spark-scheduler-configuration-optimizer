import logging

import requests
from requests.exceptions import ConnectionError
import torch

from optimizer.environment.stateinvalidexception import StateInvalidException
from optimizer.environment.stateobtaining.yarnmodel import *
from optimizer.environment.stateobtaining.stateprocessinghelper import StateProcessingHelper
from optimizer.environment.timedelayprediction import DelayPredictor
from optimizer.hyperparameters import CONTAINER_NUM_VCORES, CONTAINER_MEM
from optimizer.util import jsonutil


class StateBuilder(object):

    def __init__(self, rm_api_url: str, spark_history_server_api_url: str, scheduler_strategy):
        self.RM_API_URL = rm_api_url
        self.SPARK_HISTORY_SERVER_API_URL = spark_history_server_api_url

        self.scheduler_strategy = scheduler_strategy
        self.delay_predictor = DelayPredictor(spark_history_server_api_url)

    def build(self):
        try:
            waiting_apps = self.parse_and_build_waiting_apps()
            running_apps = self.parse_and_build_running_apps()
            constraints = self.parse_and_build_constraints()
            return State(waiting_apps, running_apps, constraints)
        except (TypeError, KeyError, ConnectionError, requests.exceptions.HTTPError) as e:
            logging.debug(e, exc_info=True)
            raise StateInvalidException

    @staticmethod
    def normalize_state_tensor(state: torch.Tensor):
        return StateProcessingHelper.normalize_state_tensor(state)

    @staticmethod
    def build_tensor(normalized_state: State):
        return StateProcessingHelper.to_tensor(normalized_state)

    def parse_and_build_running_apps(self) -> List[YarnApplication]:
        url = self.RM_API_URL + 'ws/v1/cluster/apps?states=RUNNING'
        app_json = jsonutil.get_json(url)
        return self.build_apps_from_json(app_json)

    def parse_and_build_waiting_apps(self) -> List[YarnApplication]:
        url = self.RM_API_URL + 'ws/v1/cluster/apps?states=NEW,NEW_SAVING,SUBMITTED,ACCEPTED'
        app_json = jsonutil.get_json(url)
        return self.build_apps_from_json(app_json)

    def predict_delays(self, resources, running_apps, waiting_apps):
        self.delay_predictor.predict(resources, running_apps, waiting_apps)

    def parse_and_build_finished_apps(self) -> List[FinishedApplication]:
        url = self.RM_API_URL + 'ws/v1/cluster/apps?states=FINISHED'
        app_json = jsonutil.get_json(url)
        return self.build_finished_jobs_from_json(app_json)

    def parse_and_build_constraints(self) -> List[QueueConstraint]:
        return self.scheduler_strategy.get_queue_constraints()

    @staticmethod
    def parse_and_build_queue_resources(resources, constraints) -> list:
        sum_vcore_num, sum_memory = 0, 0
        for r in resources:
            sum_vcore_num += r.vcore_num
            sum_memory += r.mem
        cluster_num_containers = min(sum_vcore_num // CONTAINER_NUM_VCORES, sum_memory // CONTAINER_MEM)

        queue_resources = []
        constraints.sort(key=lambda i: i.converted_name)
        for c in constraints:
            queue_resources.append(int(c.capacity / 100 * cluster_num_containers))

        return queue_resources

    def build_apps_from_json(self, j: dict) -> List[YarnApplication]:
        if j['apps'] is None:
            return []

        apps_json, apps = j['apps']['app'], []
        for a in apps_json:
            app_id = a['id']
            location = a['queue']
            request_container_count = self.get_request_container_count(a)
            apps.append(YarnApplication(location, request_container_count, app_id))

        return apps

    @staticmethod
    def build_finished_jobs_from_json(j: dict) -> List[FinishedApplication]:
        if j['apps'] is None:
            return []

        apps_json, apps = j['apps']['app'], []
        for a in apps_json:
            started_time = a['startedTime']
            finished_time = a['finishedTime']
            elapsed_time = a['elapsedTime']
            apps.append(FinishedApplication(started_time, finished_time, elapsed_time))

        return apps

    @staticmethod
    def get_request_container_count(j: dict) -> int:
        resource_requests = j.get('resourceRequests', [])
        return len(resource_requests)
