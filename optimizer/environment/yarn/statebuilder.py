from typing import Tuple
import logging

import requests
import torch
from requests.exceptions import ConnectionError

from optimizer.environment.spark.completedsparkapplicationanalyzer import CompletedSparkApplicationAnalyzer
from optimizer.environment.spark.sparkapplicationbuilder import SparkApplicationBuilder
from optimizer.environment.spark.sparkapplicationtimedelaypredictor import SparkApplicationTimeDelayPredictor
from optimizer.environment.stateinvalidexception import StateInvalidException
from optimizer.environment.yarn.yarnmodel import *
from optimizer.hyperparameters import STATE_SHAPE
from optimizer.util import jsonutil


class StateBuilder(object):

    def __init__(self, rm_api_url: str, spark_history_server_api_url: str, scheduler_strategy):
        self.logger = logging.getLogger(__name__)
        self.RM_API_URL = rm_api_url
        self.SPARK_HISTORY_SERVER_API_URL = spark_history_server_api_url
        self.scheduler_strategy = scheduler_strategy
        self.application_time_delay_predictor = SparkApplicationTimeDelayPredictor(spark_history_server_api_url)
        self._tmp_add_models()

    # TODO: Replace this with train set.
    def _tmp_add_models(self):
        builder = SparkApplicationBuilder(self.SPARK_HISTORY_SERVER_API_URL)
        analyzer = CompletedSparkApplicationAnalyzer()
        predictor = self.application_time_delay_predictor

        app = builder.build('application_1562834622700_0051')
        svm_model = analyzer.analyze(app)
        predictor.add_algorithm('linear', svm_model)

        # ALS
        app = builder.build('application_1562834622700_0039')
        als_model = analyzer.analyze(app)
        predictor.add_algorithm('als', als_model)

        # KMeans
        app = builder.build('application_1562834622700_0018')
        svm_model = analyzer.analyze(app)
        predictor.add_algorithm('kmeans', svm_model)

        # SVM
        app = builder.build('application_1562834622700_0014')
        svm_model = analyzer.analyze(app)
        predictor.add_algorithm('svm', svm_model)

        # Bayes
        app = builder.build('application_1562834622700_0043')
        svm_model = analyzer.analyze(app)
        predictor.add_algorithm('bayes', svm_model)

        # FPGrowth
        app = builder.build('application_1562834622700_0054')
        svm_model = analyzer.analyze(app)
        predictor.add_algorithm('FPGrowth', svm_model)

        # LDA
        app = builder.build('application_1562834622700_0058')
        svm_model = analyzer.analyze(app)
        predictor.add_algorithm('lda', svm_model)

    def build(self):
        try:
            waiting_apps, running_apps = self.parse_and_build_applications()
            resources = self.parse_and_build_resources()
            constraints = self.parse_and_build_constraints()
            return State(waiting_apps, running_apps, resources, constraints)
        except (ConnectionError, TypeError, requests.exceptions.HTTPError) as e:
            self.logger.info(e)
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
        url = self.RM_API_URL + 'ws/v1/cluster/apps?states=NEW,NEW_SAVING,SUBMITTED,ACCEPTED'
        app_json = jsonutil.get_json(url)
        return self.build_waiting_apps_from_json(app_json)

    def parse_and_build_running_apps(self) -> List[RunningApplication]:
        url = self.RM_API_URL + 'ws/v1/cluster/apps?states=RUNNING'
        app_json = jsonutil.get_json(url)
        return self.build_running_apps_from_json(app_json)

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

    def build_running_apps_from_json(self, j: dict) -> List[RunningApplication]:
        if j['apps'] is None:
            return []

        apps_json, apps = j['apps']['app'], []
        for j in apps_json:
            application_id = j['id']
            name = j['name']
            elapsed_time = j['elapsedTime']
            priority = j['priority']
            progress = j['progress']
            queue_usage_percentage = j['queueUsagePercentage']
            location = j['queue']
            predicted_time_delay = self.application_time_delay_predictor.predict(application_id, name)
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
