import logging

import requests
from requests.exceptions import ConnectionError

from optimizer.environment.stateinvalidexception import StateInvalidException
from optimizer.environment.stateobtaining.yarnmodel import *
from optimizer.environment.stateobtaining.stateprocessor import StateProcessor
from optimizer.environment.timedelayprediction import DelayPredictor
from optimizer.hyperparameters import CONTAINER_NUM_VCORES, CONTAINER_MEM
from optimizer.util import jsonutil


class StateBuilder(object):

    def __init__(self, rm_api_url: str, spark_history_server_api_url: str, scheduler_strategy):
        self.RM_API_URL = rm_api_url
        self.SPARK_HISTORY_SERVER_API_URL = spark_history_server_api_url

        self.state_processor = StateProcessor()
        self.scheduler_strategy = scheduler_strategy
        self.delay_predictor = DelayPredictor(spark_history_server_api_url)

    def build(self):
        try:
            resources = self.parse_and_build_resources()
            constraints = self.parse_and_build_constraints()
            running_apps = self.parse_and_build_running_apps()
            waiting_apps = self.parse_and_build_waiting_apps()
            queue_resources = self.parse_and_build_queue_resources(resources, constraints)
            self.predict_delays(queue_resources, running_apps, waiting_apps)
            return State(waiting_apps, running_apps, resources, constraints)
        except (TypeError, KeyError, ConnectionError, requests.exceptions.HTTPError) as e:
            logging.debug(e, exc_info=True)
            raise StateInvalidException

    def normalize_state(self, state: State):
        return self.state_processor.normalize_state(state)

    def build_tensor(self, normalized_state: State):
        return self.state_processor.to_tensor(normalized_state)

    def parse_and_build_running_apps(self) -> List[RunningApplication]:
        url = self.RM_API_URL + 'ws/v1/cluster/apps?states=RUNNING'
        app_json = jsonutil.get_json(url)
        return self.build_running_apps_from_json(app_json)

    def parse_and_build_waiting_apps(self) -> List[WaitingApplication]:
        url = self.RM_API_URL + 'ws/v1/cluster/apps?states=NEW,NEW_SAVING,SUBMITTED,ACCEPTED'
        app_json = jsonutil.get_json(url)
        return self.build_waiting_apps_from_json(app_json)

    def predict_delays(self, resources, running_apps, waiting_apps):
        self.delay_predictor.predict(resources, running_apps, waiting_apps)

    def parse_and_build_finished_apps(self) -> List[FinishedApplication]:
        url = self.RM_API_URL + 'ws/v1/cluster/apps?states=FINISHED'
        app_json = jsonutil.get_json(url)
        return self.build_finished_jobs_from_json(app_json)

    def parse_and_build_resources(self) -> List[Resource]:
        url = self.RM_API_URL + 'ws/v1/cluster/nodes'
        conf = jsonutil.get_json(url)
        nodes = conf['nodes']['node']
        resources = []
        for n in nodes:
            memory = (int(n['usedMemoryMB']) + int(n['availMemoryMB'])) / 1024
            vcores = int(n['usedVirtualCores']) + int(n['availableVirtualCores'])
            resources.append(Resource(vcores, int(memory)))
        return resources

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

    def build_running_apps_from_json(self, j: dict) -> List[RunningApplication]:
        if j['apps'] is None:
            return []

        apps_json, apps = j['apps']['app'], []
        for a in apps_json:
            app_id = a['id']
            name = a['name']
            started_time = a['startedTime']
            elapsed_time = a['elapsedTime']
            running_containers = a['runningContainers']
            priority = a['priority']
            progress = a['progress']
            queue_usage_percentage = a['queueUsagePercentage']
            location = a['queue']
            request_resources = self.build_request_resources_from_json(a)
            apps.append(RunningApplication(app_id, name, started_time, elapsed_time, running_containers, priority,
                                           location, progress, queue_usage_percentage, request_resources))

        return apps

    def build_waiting_apps_from_json(self, j: dict) -> List[WaitingApplication]:
        if j['apps'] is None:
            return []

        apps_json, apps = j['apps']['app'], []
        for a in apps_json:
            application_id = a['id']
            name = a['name']
            started_time = a['startedTime']
            elapsed_time = a['elapsedTime']
            priority = a['priority']
            location = a['queue']
            request_resources = self.build_request_resources_from_json(a)
            apps.append(WaitingApplication(application_id, name, started_time,
                                           elapsed_time, priority, location, request_resources))

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
    def build_request_resources_from_json(j: dict) -> List[ApplicationRequestResource]:
        ret = []

        if 'resourceRequests' not in j:
            return ret

        resource_requests = j['resourceRequests']
        for req in resource_requests:
            priority = req['priority']
            capability = req['capability']
            memory = capability['memory']
            cpu = capability['vCores']
            ret.append(ApplicationRequestResource(priority, memory, cpu))

        return ret
