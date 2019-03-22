import json
from typing import Dict, List

import requests
import torch

from .communicator import Communicator


class Job(object):
    def __init__(self, submit_time, priority, tasks):
        self.submit_time: int = submit_time
        self.priority: int = priority
        self.tasks: List[Task] = tasks


class Task(object):
    def __init__(self, platform, start_time, end_time):
        self.platform: str = platform
        self.start_time: int = start_time
        self.end_time: int = end_time


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
    def __init__(self, job_name, host_node_name):
        self.job_name = job_name
        self.host_node_name = host_node_name


class Constraint(object):
    def __init__(self):
        self.queue: List[QueueConstraint] = []
        self.job = []

    def add_queue_c(self, c: QueueConstraint):
        self.queue.append(c)


class State(object):
    def __init__(self, jobs: List[Job], resources: List[Resource], constraint: Constraint):
        self.jobs = jobs
        self.resources = resources
        self.constraint = constraint


class Action(object):
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


class YarnSchedulerCommunicator(Communicator):
    """
    Uses RESTFul API to communicate with YARN cluster scheduler.
    """

    def __init__(self, rm_host: str, hadoop_home: str):
        self.hadoop_etc = hadoop_home + '/etc'
        self.rm_host = rm_host
        self.state = self.get_state()

    @staticmethod
    def get_action_set() -> Dict[int, Action]:
        """
        :return: Action dictionary defined in document.
        """
        return {
            1: Action(1, 5),
            2: Action(2, 4),
            3: Action(3, 3),
            4: Action(4, 2),
            5: Action(5, 1)
        }

    def act(self, action: int) -> float:
        """
        Apply action and see how many rewards we can get.
        :return: Reward this step got.
        """
        self.save_conf()
        return 0

    def get_state(self) -> State:
        """
        Get raw state of YARN.
        """
        jobs = self.__get_jobs()
        resources = self.__get_resources()
        constraints = self.__get_constraints()
        return State(jobs, resources, constraints)

    def get_state_tensor(self) -> torch.Tensor:
        """
        Get state of YARN which is trimmed.
        Which is defined as the ϕ(s) function defined in document.
        """
        raw_state = self.get_state()
        return torch.Tensor()

    def save_conf(self) -> None:
        """
        Save configurations by using RESTFul API.
        """
        pass

    def is_done(self) -> bool:
        """
        Test if all jobs are done.
        """
        pass

    def close(self) -> None:
        """
        Close the communication.
        """
        pass

    def reset(self) -> None:
        """
        Resets the env in order to run this program again.
        """
        pass

    def __get_jobs(self) -> List[Job]:
        url = self.rm_host + 'ws/v1/cluster/apps'
        # conf = get_json(url)
        conf = get_json_test('/Users/xenon/Desktop/SLS/cluster-apps.json')
        apps = conf['apps']['app']
        jobs = []
        for j in apps:
            if j['state'] != 'WAITING':
                continue
            start_time = j['startedTime']
            priority = j['priority']
            jobs.append(Job(start_time, priority, []))
        return jobs

    def __get_resources(self) -> List[Resource]:
        url = self.rm_host + 'ws/v1/cluster/nodes'
        # conf = get_json(url)
        conf = get_json_test('/Users/xenon/Desktop/SLS/cluster-nodes.json')
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
        # conf = get_json(url)
        conf = get_json_test('/Users/xenon/Desktop/SLS/cluster-scheduler.json')
        constraint = Constraint()

        queues = conf['scheduler']['schedulerInfo']['queues']['queue']
        for q in queues:
            capacity = q['capacity']
            max_capacity = q['maxCapacity']
            name = q['queueName']
            queue_c = QueueConstraint(name, capacity, max_capacity)
            constraint.add_queue_c(queue_c)

        return constraint


def get_json(url: str) -> Dict[str, object]:
    r = requests.get(url)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    j = json.loads(r.text)
    return j


def get_json_test(filename: str) -> Dict[str, object]:
    with open(filename) as f:
        return json.load(f)
