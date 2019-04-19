import json
import math
import os
import subprocess
import time
from typing import Dict, List, Tuple, Union

import pandas as pd
import requests
import torch
from requests.exceptions import ConnectionError

from .communicator import Communicator
from .exceptions import StateInvalidException
from ..hyperparameters import STATE_SHAPE
from ..xmlutil import XmlModifier


class Job(object):
    def __init__(self, application_id, submit_time, wait_time, priority, tasks):
        self.application_id = application_id
        self.submit_time: int = submit_time
        self.wait_time: int = wait_time
        self.priority: int = priority
        self.tasks: List[Task] = tasks

    def __eq__(self, other):
        return self.application_id == other.application_id

    # Use application_id to identity a Job
    def __hash__(self):
        return hash(self.application_id)


class Task(object):
    def __init__(self, platform, memory, cpu):
        self.platform: str = platform
        self.memory: int = memory
        self.cpu: int = cpu


class Resource(object):
    def __init__(self, platform: str, cpu: int, mem: int):
        self.node_name: str = platform   # Node Name
        self.cpu: int = cpu              # CPU Cores
        self.mem: int = mem              # Memory(GB)


class QueueConstraint(object):
    def __init__(self, name, capacity=100, max_capacity=100):
        self.name = name
        self.capacity = capacity
        self.max_capacity = max_capacity


class JobConstraint(object):
    def __init__(self, job_id, host_node_name):
        self.job_id = job_id
        self.host_node_name = host_node_name


class Constraint(object):
    def __init__(self):
        self.queue: List[QueueConstraint] = []
        self.job = []

    def add_queue_c(self, c: QueueConstraint):
        self.queue.append(c)

    def add_job_c(self, c: JobConstraint):
        self.job.append(c)


class State(object):
    def __init__(self, waiting_jobs: List[Job], running_jobs: List[Job],
                 resources: List[Resource], constraint: Constraint):
        self.awaiting_jobs = waiting_jobs
        self.running_jobs: List[Job] = running_jobs
        self.resources = resources
        self.constraint = constraint


class Action(object):
    def __init__(self, a: int, b: int):
        self.queue_a_weight = a
        self.queue_b_weight = b


class YarnCommunicator(Communicator):
    """
    Uses RESTFul API to communicate with YARN cluster scheduler.
    """

    def __init__(self, rm_host: str, hadoop_home: str, sls_jobs_json: str):
        self.hadoop_home = hadoop_home
        self.hadoop_etc = hadoop_home + '/etc/hadoop'
        self.rm_host = rm_host
        self.sls_jobs_json = sls_jobs_json

        self.awaiting_jobs: List[Job] = []

        self.action_set = self.get_action_set()
        self.sls_runner: subprocess.Popen = None

        self.copy_conf_file()

    @staticmethod
    def get_action_set() -> Dict[int, Action]:
        """
        :return: Action dictionary defined in document.
        """
        return {
            0: Action(3, 3),
            1: Action(1, 5),
            2: Action(2, 4),
            3: Action(4, 2),
            4: Action(5, 1)
        }

    def override_scheduler_xml_with(self, action_index: int):
        """
        Override capacity-scheduler.xml or fair-scheduler-template.xml
        to change the capacity or weight of queues.
        """
        dest = self.hadoop_etc + '/capacity-scheduler.xml'
        xml_modifier = XmlModifier('./data/capacity-scheduler-template.xml', dest)

        a, b = self.__get_capacities_by_action_index(action_index)

        xml_modifier.modify('yarn.scheduler.capacity.root.queueA.capacity', a)
        xml_modifier.modify('yarn.scheduler.capacity.root.queueB.capacity', b)
        xml_modifier.modify('yarn.scheduler.capacity.root.queueA.maximum-capacity', a)
        xml_modifier.modify('yarn.scheduler.capacity.root.queueB.maximum-capacity', b)

        xml_modifier.save()

    def act(self, action_index: int) -> float:
        """
        Apply action and see how many rewards we can get.
        :return: Reward this step gets.
        """
        self.set_queue_weights(action_index)
        return self.get_reward()

    def get_reward(self) -> float:
        job_count = len(self.awaiting_jobs)
        if job_count == 0:
            return 0

        total_wait_time = 0.0
        for j in self.awaiting_jobs:
            total_wait_time += j.wait_time
        average_wait_time = total_wait_time / job_count

        # TODO: Try -math.tanh(RATE * (average_wait_time - BIAS))
        cost = math.tanh(0.00001 * average_wait_time)
        print('\n', 'Cost:', cost)
        return -cost

    def get_state(self) -> State:
        """
        Get raw state of YARN.
        """
        try:
            wj, rj = self.__get_jobs()
            resources = self.__get_resources()
            constraints = self.__get_constraints()
            self.awaiting_jobs = wj
            return State(wj, rj, resources, constraints)
        except ConnectionError:
            raise StateInvalidException()
        except TypeError:
            raise StateInvalidException()
        except requests.exceptions.HTTPError:
            raise StateInvalidException()

    def get_state_tensor(self) -> Union[torch.Tensor, None]:
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
        tensor = torch.zeros(14, 126)

        # Line 0-4: awaiting jobs and their tasks
        for i, wj in enumerate(raw.awaiting_jobs[:5]):
            idx = 0
            tensor[i][idx] = wj.submit_time
            idx += 1
            tensor[i][idx] = wj.priority
            idx += 1
            for t in wj.tasks:
                tensor[i][idx] = t.memory
                idx += 1
                tensor[i][idx] = t.cpu
                idx += 1

        # Line 5-9: running jobs and their tasks
        for i, rj in enumerate(raw.running_jobs[:5]):
            idx = 0
            row = i + 5
            tensor[row][idx] = rj.submit_time
            idx += 1
            tensor[row][idx] = rj.priority
            idx += 1
            for t in rj.tasks:
                tensor[row][idx] = t.memory
                idx += 1
                tensor[row][idx] = t.cpu
                idx += 1

        # Line 10: resources of cluster
        row, idx = 10, 0
        for r in raw.resources:
            tensor[row][idx] = r.mem
            idx += 1
            tensor[row][idx] = r.cpu
            idx += 1

        # Line 11: queue constraints
        row, idx = 11, 0
        for c in raw.constraint.queue:
            tensor[row][idx] = c.max_capacity
            idx += 1
            tensor[row][idx] = c.capacity
            idx += 1

        # Line 12: job constraints(leave empty for now)
        for _ in raw.constraint.job:
            pass

        return tensor.reshape(*STATE_SHAPE)

    def set_queue_weights(self, action_index: int) -> None:
        """
        Use script "refresh-queues.sh" to refresh queues' configurations.
        """
        self.override_scheduler_xml_with(action_index)

        wd = os.getcwd()
        cmd = [wd + '/bin/refresh-queues.sh', self.hadoop_home]
        subprocess.Popen(cmd)

    def is_done(self) -> bool:
        """
        Test if all jobs are done.
        """
        return self.sls_runner is None or self.sls_runner.poll() is not None

    def close(self) -> None:
        """
        Close the communication.
        """
        if self.sls_runner is not None:
            self.sls_runner.terminate()
            self.sls_runner.wait()
            time.sleep(5)
            self.sls_runner = None

    def reset(self, wd=None) -> None:
        """
        Resets the env in order to run this program again.
        """
        if wd is None:
            wd = os.getcwd()

        self.close()

        sls_jobs_json = wd + '/' + self.sls_jobs_json
        cmd = "{}/bin/start-sls.sh {} {} {}".format(wd, self.hadoop_home, wd, sls_jobs_json)
        self.sls_runner = subprocess.Popen(cmd.split(' '))

        # Wait until web server starts.
        time.sleep(20)

    def copy_conf_file(self):
        with open('./data/capacity-scheduler.xml', 'r') as src:
            with open(self.hadoop_etc + '/capacity-scheduler.xml', 'w') as dest:
                dest.write(src.read())

    def get_total_time_cost(self):
        data = pd.read_csv('./results/logs/jobruntime.csv')
        end_time = data['simulate_end_time']
        start_time = data['simulate_start_time']
        time_costs = end_time - start_time
        sum_time_cost = time_costs.sum()
        return time_costs, sum_time_cost

    def __get_capacities_by_action_index(self, action_index: int) -> Tuple[float, float]:
        action = self.action_set[action_index]
        weight_a = action.queue_a_weight
        weight_b = action.queue_b_weight
        total_weight = weight_a + weight_b
        a = 100 * weight_a / total_weight
        b = 100 * weight_b / total_weight
        return a, b

    def __get_jobs(self) -> Tuple[List[Job], List[Job]]:
        running_jobs = self.__get_running_jobs()
        waiting_jobs = self.__get_waiting_jobs()
        return waiting_jobs, running_jobs

    def __get_jobs_by_states(self, states: str) -> List[Job]:
        url = self.rm_host + 'ws/v1/cluster/apps?states=' + states
        job_json = get_json(url)
        return self.__build_jobs_from_json(job_json)

    @staticmethod
    def __build_jobs_from_json(j: Dict[str, object]) -> List[Job]:
        if j['apps'] is None:
            return []

        apps = j['apps']['app']
        jobs = []
        for j in apps:
            application_id = j['id']
            start_time = j['startedTime']
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

    # TODO: Get wait time of these jobs in order to estimate the effect of this algorithm.
    def __get_waiting_jobs(self) -> List[Job]:
        return self.__get_jobs_by_states('NEW,NEW_SAVING,SUBMITTED,ACCEPTED')

    def __get_running_jobs(self) -> List[Job]:
        return self.__get_jobs_by_states('RUNNING')

    def __get_resources(self) -> List[Resource]:
        url = self.rm_host + 'ws/v1/cluster/nodes'
        conf = get_json(url)
        nodes = conf['nodes']['node']
        resources = []
        for n in nodes:
            node_name = n['nodeHostName']
            memory = (int(n['usedMemoryMB']) + int(n['availMemoryMB'])) / 1024
            vcores = int(n['usedVirtualCores']) + int(n['availableVirtualCores'])
            resources.append(Resource(node_name, vcores, int(memory)))
        return resources

    def __get_constraints(self) -> Constraint:
        url = self.rm_host + 'ws/v1/cluster/scheduler'
        conf = get_json(url)
        constraint = Constraint()

        queues = conf['scheduler']['schedulerInfo']['queues']['queue']
        for q in queues:
            capacity = q['capacity']
            max_capacity = q['maxCapacity']
            name = q['queueName']
            queue_c = QueueConstraint(name, capacity, max_capacity)
            constraint.add_queue_c(queue_c)

        # TODO LATER: Job constraint
        for job in self.awaiting_jobs:
            try:
                url = "{}ws/v1/cluster/apps/{}/appattempts".format(self.rm_host, job.application_id)
                attempt_json = get_json(url)
                attempts = attempt_json['appAttempts']['appAttempt']
                for a in attempts:
                    http_address = a['nodeHttpAddress']
                    if len(http_address):
                        constraint.add_job_c(JobConstraint(job.application_id, http_address))
            except requests.exceptions.HTTPError:
                pass

        return constraint


def get_current_time_ms() -> int:
    return int(time.time() * 1000)


def get_json(url: str) -> Dict[str, object]:
    r = requests.get(url)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    j = json.loads(r.text)
    return j
