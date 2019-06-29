import abc
import os
import subprocess
import time
from typing import Dict, Tuple

import math
import requests
import torch
from requests.exceptions import ConnectionError

from optimizer.hyperparameters import STATE_SHAPE
from .actionparser import ActionParser
from .iresetablecommunicator import ICommunicator
from .schedulerstrategy import SchedulerStrategyFactory
from .yarnmodel import *
from ..stateinvalidexception import StateInvalidException
from ...util import jsonutil


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
        self.awaiting_jobs: List[Job] = []

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
        job_count = len(self.awaiting_jobs)
        if job_count == 0:
            return 0

        total_wait_time = 0.0
        for j in self.awaiting_jobs:
            total_wait_time += j.wait_time

        average_wait_time = total_wait_time / job_count
        cost = math.tanh(0.00001 * average_wait_time)
        return -cost

    def get_state(self) -> State:
        """
        Get raw state of YARN.
        """
        try:
            wj, rj = self._get_jobs()
            resources = self._get_resources()
            constraints = self._get_constraints()
            self.awaiting_jobs = wj
            return State(wj, rj, resources, constraints)
        except ConnectionError:
            raise StateInvalidException
        except TypeError:
            raise StateInvalidException
        except requests.exceptions.HTTPError:
            raise StateInvalidException

    def get_state_tensor(self) -> torch.Tensor:
        """
        Get state of YARN which is trimmed.
        Which is defined as the ϕ(s) function defined in document.

        state: {
            awaiting_jobs: [Job, Job, ...],
            running_jobs: [Job, Job, ...],
            resources: [Resource, Resource, ...],
            constraint: {
                queue: [QueueConstraint, QueueConstraint, ...],
                job: [JobConstraint, JobConstraint, ...],
            }
        }
        """
        raw = self.get_state()
        height, width = STATE_SHAPE
        tensor = torch.zeros(height, width)

        # Line 0-74: awaiting jobs and their tasks
        for i, wj in enumerate(raw.awaiting_jobs[:75]):
            idx = 0
            tensor[i][idx] = wj.submit_time
            idx += 1
            tensor[i][idx] = wj.priority
            idx += 1
            for t in wj.tasks[:199]:
                tensor[i][idx] = t.memory
                idx += 1
                tensor[i][idx] = t.cpu
                idx += 1

        # Line 75-149: running jobs and their tasks
        for i, rj in enumerate(raw.running_jobs[:75]):
            idx = 0
            row = i + 75
            tensor[row][idx] = rj.submit_time
            idx += 1
            tensor[row][idx] = rj.priority
            idx += 1
            for t in rj.tasks[:199]:
                tensor[row][idx] = t.memory
                idx += 1
                tensor[row][idx] = t.cpu
                idx += 1

        # Line 150-197: resources of cluster
        row, idx = 150, 0
        for r in raw.resources[:4800]:
            tensor[row][idx] = r.mem
            idx += 1
            tensor[row][idx] = r.cpu
            idx += 1
            if idx == 200:
                row += 1
                idx = 0

        # Line 198: queue constraints
        row, idx = 198, 0
        for c in raw.constraint.queue[:100]:
            tensor[row][idx] = c.max_capacity
            idx += 1
            tensor[row][idx] = c.capacity
            idx += 1

        # Line 199: job constraints(leave empty for now)
        for _ in raw.constraint.job:
            pass

        return tensor

    def set_and_refresh_queue_config(self, action_index: int) -> None:
        """
        Use script "refresh-queues.sh" to refresh the configurations of queues.
        """
        self.scheduler_strategy.override_config(action_index)
        refresh_queues(self.hadoop_home)

    @abc.abstractmethod
    def is_done(self) -> bool:
        """
        Test if all jobs are done.
        """
        pass

    @abc.abstractmethod
    def get_scheduler_type(self) -> str:
        pass

    def override_config(self, action_index: int):
        self.scheduler_strategy.override_config(action_index)

    def _get_jobs(self) -> Tuple[List[Job], List[Job]]:
        running_jobs = self._get_running_jobs()
        awaiting_jobs = self._get_waiting_jobs()
        return awaiting_jobs, running_jobs

    def _get_jobs_by_states(self, states: str) -> List[Job]:
        url = self.api_url + 'ws/v1/cluster/apps?states=' + states
        job_json = jsonutil.get_json(url)
        return self._build_jobs_from_json(job_json)

    def _build_jobs_from_json(self, j: Dict[str, object]) -> List[Job]:
        if j['apps'] is None:
            return []

        apps = j['apps']['app']
        jobs = []
        for j in apps:
            application_id = j['id']
            start_time_ms = j['startedTime'] - self.start_time
            start_time = int(start_time_ms / 1000)
            wait_time = j['elapsedTime']
            priority = j['priority']
            tasks = []

            if 'resourceRequests' in j:
                resource_requests = j['resourceRequests']
                for req in resource_requests:
                    capability = req['capability']
                    memory = capability['memory']
                    cpu = capability['vCores']
                    tasks.append(Task('', memory, cpu))

            jobs.append(Job(application_id, start_time, wait_time, priority, tasks))

        return jobs

    def _get_waiting_jobs(self) -> List[Job]:
        return self._get_jobs_by_states('NEW,NEW_SAVING,SUBMITTED,ACCEPTED')

    def _get_running_jobs(self) -> List[Job]:
        return self._get_jobs_by_states('RUNNING')

    def _get_resources(self) -> List[Resource]:
        url = self.api_url + 'ws/v1/cluster/nodes'
        conf = jsonutil.get_json(url)
        nodes = conf['nodes']['node']
        resources = []
        for n in nodes:
            node_name = n['nodeHostName']
            memory = (int(n['usedMemoryMB']) + int(n['availMemoryMB'])) / 1024
            vcores = int(n['usedVirtualCores']) + int(n['availableVirtualCores'])
            resources.append(Resource(node_name, vcores, int(memory)))
        return resources

    def _get_constraints(self) -> Constraint:
        constraint = Constraint()
        self.scheduler_strategy.get_queue_constraints(constraint)
        self._get_job_constraints(constraint)
        return constraint

    def _get_job_constraints(self, constraint: Constraint):
        for job in self.awaiting_jobs:
            try:
                url = "{}ws/v1/cluster/apps/{}/appattempts".format(self.api_url, job.application_id)
                attempt_json = jsonutil.get_json(url)
                attempts = attempt_json['appAttempts']['appAttempt']
                for a in attempts:
                    http_address = a['nodeHttpAddress']
                    if len(http_address):
                        constraint.add_job_c(JobConstraint(job.application_id, http_address))
            except requests.exceptions.HTTPError:
                pass


def timestamp() -> int:
    return int(time.time() * 1000)


def refresh_queues(hadoop_home: str):
    subprocess.Popen([os.path.join(os.getcwd(), 'bin', 'refresh-queues.sh'), hadoop_home])
