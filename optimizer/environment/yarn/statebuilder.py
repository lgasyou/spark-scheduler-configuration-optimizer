from typing import Tuple

import requests
import torch
from requests.exceptions import ConnectionError

from optimizer.hyperparameters import STATE_SHAPE
from .yarnmodel import *
from optimizer.environment.spark.sparkapplicationtimedelaypredictor import SparkApplicationTimeDelayPredictor
from optimizer.environment.stateinvalidexception import StateInvalidException
from optimizer.util import jsonutil


class StateBuilder(object):

    def __init__(self, rm_api_url: str, spark_history_server_api_url: str, scheduler_strategy):
        self.rm_api_url = rm_api_url
        self.spark_history_server_api_url = spark_history_server_api_url
        self.scheduler_strategy = scheduler_strategy
        self.application_time_delay_predictor = SparkApplicationTimeDelayPredictor(spark_history_server_api_url)

    def build(self):
        try:
            waiting_apps, running_apps = self.parse_and_build_applications()
            resources = self.parse_and_build_resources()
            constraints = self.parse_and_build_constraints()
            return State(waiting_apps, running_apps, resources, constraints)
        except (ConnectionError, TypeError, requests.exceptions.HTTPError):
            raise StateInvalidException

    @staticmethod
    def build_tensor(raw: State):
        height, width = STATE_SHAPE
        tensor = torch.zeros(height, width)

        # Line 0-74: waiting apps and their resource requests
        for i, wa in enumerate(raw.waiting_apps[:75]):
            line = [wa.elapsed_time, wa.priority, wa.converted_location]
            for rr in wa.request_resources[:64]:
                line.extend([rr.priority, rr.memory, rr.cpu])
            line.extend([0.0] * (width - len(line)))
            tensor[i] = torch.Tensor(line)

        # Line 75-149: running apps and their resource requests
        for i, ra in enumerate(raw.running_apps[:75]):
            row = i + 75
            line = [ra.elapsed_time, ra.priority, ra.converted_location,
                    ra.progress, ra.queue_usage_percentage, ra.predicted_time_delay]
            for rr in ra.request_resources[:65]:
                line.extend([rr.priority, rr.memory, rr.cpu])
            line.extend([0.0] * (width - len(line)))
            tensor[row] = torch.Tensor(line)

        # Line 150-198: resources of cluster
        row, idx = 150, 0
        for r in raw.resources[:4900]:
            tensor[row][idx] = r.mem
            idx += 1
            tensor[row][idx] = r.vcore_num
            idx += 1
            if idx == width:
                row += 1
                idx = 0

        # Line 199: queue constraints
        row, queue_constraints = 199, []
        for c in raw.constraints[:50]:
            queue_constraints.extend([c.converted_name, c.capacity, c.max_capacity, c.used_capacity])
        queue_constraints.extend([0.0] * (width - len(queue_constraints)))
        tensor[row] = torch.Tensor(queue_constraints)

        return tensor

    def parse_and_build_applications(self) -> Tuple[List[WaitingApplication], List[RunningApplication]]:
        waiting_apps = self.parse_and_build_waiting_apps()
        running_apps = self.parse_and_build_running_apps()
        return waiting_apps, running_apps

    def parse_and_build_waiting_apps(self) -> List[WaitingApplication]:
        url = self.rm_api_url + 'ws/v1/cluster/apps?states=NEW,NEW_SAVING,SUBMITTED,ACCEPTED'
        app_json = jsonutil.get_json(url)
        return self.build_waiting_apps_from_json(app_json)

    def parse_and_build_running_apps(self) -> List[RunningApplication]:
        url = self.rm_api_url + 'ws/v1/cluster/apps?states=RUNNING'
        app_json = jsonutil.get_json(url)
        return self.build_running_apps_from_json(app_json)

    def parse_and_build_resources(self) -> List[Resource]:
        url = self.rm_api_url + 'ws/v1/cluster/nodes'
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

    def build_running_apps_from_json(self, j: dict) -> List[RunningApplication]:
        if j['apps'] is None:
            return []

        apps_json, apps = j['apps']['app'], []
        for j in apps_json:
            application_id = j['id']
            elapsed_time = j['elapsedTime']
            priority = j['priority']
            progress = j['progress']
            queue_usage_percentage = j['queueUsagePercentage']
            location = j['queue']
            predicted_time_delay, _ = self.application_time_delay_predictor.predict(application_id)
            request_resources = self.build_request_resources_from_json(j)
            apps.append(RunningApplication(application_id, elapsed_time, priority, location, progress,
                                           queue_usage_percentage, predicted_time_delay, request_resources))

        return apps

    def build_waiting_apps_from_json(self, j: dict) -> List[WaitingApplication]:
        if j['apps'] is None:
            return []

        apps_json = j['apps']['app']
        apps = []
        for j in apps_json:
            elapsed_time = j['elapsedTime']
            priority = j['priority']
            location = j['queue']
            request_resources = self.build_request_resources_from_json(j)
            apps.append(WaitingApplication(elapsed_time, priority, location, request_resources))

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