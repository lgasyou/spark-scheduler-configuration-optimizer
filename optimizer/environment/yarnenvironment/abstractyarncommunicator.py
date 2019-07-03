import abc
import os
import subprocess
import time
from typing import Dict, Tuple, Optional

import math
import requests
import torch
from requests.exceptions import ConnectionError

from optimizer.hyperparameters import STATE_SHAPE, QUEUES
from .actionparser import ActionParser
from .iresetablecommunicator import ICommunicator
from .schedulerstrategy import SchedulerStrategyFactory
from .yarnmodel import *
from ..stateinvalidexception import StateInvalidException
from optimizer.util import jsonutil


class AbstractYarnCommunicator(ICommunicator):
    """
    Uses RESTFul API to communicate with YARN cluster scheduler.
    """

    def __init__(self, api_url: str, hadoop_home: str):
        self.hadoop_home = hadoop_home
        self.hadoop_etc = hadoop_home + '/etc/hadoop'
        self.api_url = api_url

        self.start_time = timestamp()
        self.action_set = ActionParser.parse()
        self.state: Optional[State] = None

        scheduler_type = self.get_scheduler_type()
        self.scheduler_strategy = SchedulerStrategyFactory.create(scheduler_type, api_url,
                                                                  self.hadoop_etc, self.action_set)
        self.scheduler_strategy.copy_conf_file()

    def act(self, action_index: int) -> float:
        """
        Apply action and see how many rewards we can get.
        :return: Reward this step gets.
        """
        self.set_and_refresh_queue_config(action_index)
        return self.get_reward()

    def get_reward(self) -> float:
        waiting_jobs = self.state.waiting_jobs
        running_jobs = self.state.running_jobs
        job_count = len(waiting_jobs) + len(running_jobs)
        if job_count == 0:
            return 0

        waiting_jobs_elapsed_time = sum([job.elapsed_time for job in waiting_jobs])
        running_jobs_elapsed_time = sum([job.elapsed_time for job in running_jobs])
        middle = (0.8 * waiting_jobs_elapsed_time + 0.2 * running_jobs_elapsed_time) / job_count
        cost = math.tanh(0.00001 * middle)
        return -cost

    def get_state(self) -> State:
        """
        Get raw state of YARN.
        """
        try:
            wj, rj = self._get_jobs()
            resources = self._get_resources()
            constraints = self._get_constraints()
            self.state = State(wj, rj, resources, constraints)
            return self.state
        except (ConnectionError, TypeError, requests.exceptions.HTTPError):
            raise StateInvalidException

    def get_state_tensor(self) -> torch.Tensor:
        """
        Get state of YARN which is trimmed.
        Which is defined as the ϕ(s) function defined in document.

        state: {
            waiting_jobs: [WaitingJob, WaitingJob, ...],
            running_jobs: [RunningJob, RunningJob, ...],
            resources: [Resource, Resource, ...],
            queue_constraints: [QueueConstraint, QueueConstraint, ...]
        }
        """
        raw = self.get_state()
        height, width = STATE_SHAPE
        tensor = torch.zeros(height, width)

        # Line 0-74: waiting jobs and their tasks
        for i, wj in enumerate(raw.waiting_jobs[:75]):
            line = [wj.elapsed_time, wj.priority, queue_name_to_index(wj.location)]
            for rr in wj.request_resources[:64]:
                line.extend([rr.priority, rr.memory, rr.cpu])
            line.extend([0.0] * (width - len(line)))
            tensor[i] = torch.Tensor(line)

        # Line 75-149: running jobs and their tasks
        for i, rj in enumerate(raw.running_jobs[:75]):
            row = i + 75
            line = [rj.elapsed_time, rj.priority, queue_name_to_index(rj.location),
                    rj.progress, rj.queue_usage_percentage,
                    rj.memory_seconds, rj.vcore_seconds]
            for rr in rj.request_resources[:65]:
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
            queue_constraints.extend([queue_name_to_index(c.name), c.capacity, c.max_capacity, c.used_capacity])
        queue_constraints.extend([0.0] * (width - len(queue_constraints)))
        tensor[row] = torch.Tensor(queue_constraints)

        return tensor

    def set_and_refresh_queue_config(self, action_index: int) -> None:
        """
        Use script "refresh-queues.sh" to refresh the configurations of queues.
        """
        self.scheduler_strategy.override_config(action_index)
        refresh_queues(self.hadoop_home)

    @abc.abstractmethod
    def is_done(self) -> bool:
        """Test if all jobs are done."""
        pass

    @abc.abstractmethod
    def get_scheduler_type(self) -> str:
        pass

    def override_config(self, action_index: int):
        self.scheduler_strategy.override_config(action_index)

    def _get_jobs(self) -> Tuple[List[WaitingJob], List[RunningJob]]:
        waiting_jobs = self._get_waiting_jobs()
        running_jobs = self._get_running_jobs()
        return waiting_jobs, running_jobs

    def _get_waiting_jobs(self) -> List[WaitingJob]:
        url = self.api_url + 'ws/v1/cluster/apps?states=NEW,NEW_SAVING,SUBMITTED,ACCEPTED'
        job_json = jsonutil.get_json(url)
        return build_waiting_jobs_from_json(job_json)

    def _get_running_jobs(self) -> List[RunningJob]:
        url = self.api_url + 'ws/v1/cluster/apps?states=RUNNING'
        job_json = jsonutil.get_json(url)
        return build_running_jobs_from_json(job_json)

    def _get_resources(self) -> List[Resource]:
        url = self.api_url + 'ws/v1/cluster/nodes'
        conf = jsonutil.get_json(url)
        nodes = conf['nodes']['node']
        resources = []
        for n in nodes:
            memory = (int(n['usedMemoryMB']) + int(n['availMemoryMB'])) / 1024
            vcores = int(n['usedVirtualCores']) + int(n['availableVirtualCores'])
            resources.append(Resource(vcores, int(memory)))
        return resources

    def _get_constraints(self) -> List[QueueConstraint]:
        return self.scheduler_strategy.get_queue_constraints()


def timestamp() -> int:
    return int(time.time() * 1000)


def refresh_queues(hadoop_home: str):
    subprocess.Popen([os.path.join(os.getcwd(), 'bin', 'refresh-queues.sh'), hadoop_home])


def build_running_jobs_from_json(j: dict) -> List[RunningJob]:
    if j['apps'] is None:
        return []

    apps, jobs = j['apps']['app'], []
    for j in apps:
        elapsed_time = j['elapsedTime']
        priority = j['priority']
        progress = j['progress']
        queue_usage_percentage = j['queueUsagePercentage']
        memory_seconds = j['memorySeconds']
        vcore_seconds = j['vcoreSeconds']
        location = j['queue']
        request_resources = build_request_resources_from_json(j)
        jobs.append(RunningJob(elapsed_time, priority, location, progress, queue_usage_percentage, memory_seconds,
                               vcore_seconds, request_resources))

    return jobs


def build_waiting_jobs_from_json(j: dict) -> List[WaitingJob]:
    if j['apps'] is None:
        return []

    apps = j['apps']['app']
    jobs = []
    for j in apps:
        elapsed_time = j['elapsedTime']
        priority = j['priority']
        location = j['queue']
        request_resources = build_request_resources_from_json(j)
        jobs.append(WaitingJob(elapsed_time, priority, location, request_resources))

    return jobs


def build_request_resources_from_json(j: dict) -> List[JobRequestResource]:
    ret = []

    if 'resourceRequests' not in j:
        return ret

    resource_requests = j['resourceRequests']
    for req in resource_requests:
        priority = req['priority']
        capability = req['capability']
        memory = capability['memory']
        cpu = capability['vCores']
        ret.append(JobRequestResource(priority, memory, cpu))

    return ret


def queue_name_to_index(queue_name: str) -> int:
    return QUEUES['names'].index(queue_name)
