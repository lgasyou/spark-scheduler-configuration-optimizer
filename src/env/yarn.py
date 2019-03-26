import json
import os
import subprocess
import time
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
        self.queue_a_weight = a
        self.queue_b_weight = b


class YarnSchedulerCommunicator(Communicator):
    """
    Uses RESTFul API to communicate with YARN cluster scheduler.
    """

    def __init__(self, rm_host: str, hadoop_home: str):
        self.hadoop_home = hadoop_home
        self.hadoop_etc = hadoop_home + '/etc'
        self.rm_host = rm_host

        self.action_set = self.get_action_set()
        self.state: State = None
        self.sls_runner: subprocess.Popen = None

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

    # TODO: reward
    def act(self, action_index: int) -> float:
        """
        Apply action and see how many rewards we can get.
        :return: Reward this step got.
        """
        action = self.action_set[action_index]
        self.set_queue_weights(action.queue_a_weight, action.queue_b_weight)
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

    def set_queue_weights(self, queue_a_weight: int, queue_b_weight: int) -> None:
        """
        Set queue weights by using RESTFul API.
        """

        conf = """
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <sched-conf>
          <update-queue>
            <queue-name>root.a</queue-name>
            <params>
              <entry>
                <key>capacity</key>
                <value>{}</value>
              </entry>
            </params>
          </update-queue>
          <update-queue>
            <queue-name>root.b</queue-name>
            <params>
              <entry>
                <key>capacity</key>
                <value>{}</value>
              </entry>
            </params>
          </update-queue>
        </sched-conf>
        """.format(queue_a_weight, queue_b_weight)
        url = self.rm_host + 'ws/v1/cluster/scheduler-conf'

        r = requests.put(url, conf, headers={
            'Content-Type': 'application/xml'
        })
        r.raise_for_status()

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

    def reset(self) -> None:
        """
        Resets the env in order to run this program again.
        """
        self.close()
        wd = os.getcwd()
        self.sls_runner = subprocess.Popen([wd + '/start-sls.sh', self.hadoop_home, wd])
        time.sleep(5)

    def __get_jobs(self) -> List[Job]:
        waiting_jobs = self.__get_waiting_jobs()
        allocated_jobs = self.__get_allocated_jobs()
        running_jobs = self.__get_running_jobs()
        arrived_jobs = self.__get_arrived_jobs()
        completed_jobs = self.__get_completed_jobs()

        return []

    def __get_jobs_by_states(self, states: str) -> Dict[str, object]:
        url = self.rm_host + 'ws/v1/cluster/apps?states=' + states
        return get_json(url)

    @staticmethod
    def __build_jobs_from_json(j: Dict[str, object]) -> List[Job]:
        if j['apps'] is None:
            return []

        apps = j['apps']['app']
        jobs = []
        for j in apps:
            start_time = j['startedTime']
            priority = j['priority']
            jobs.append(Job(start_time, priority, []))
        return jobs

    def __get_waiting_jobs(self) -> List[Job]:
        j = self.__get_jobs_by_states('NEW,NEW_SAVING,SUBMITTED,ACCEPTED')
        return self.__build_jobs_from_json(j)

    def __get_allocated_jobs(self) -> List[Job]:
        j = self.__get_jobs_by_states('ACCEPTED')
        return self.__build_jobs_from_json(j)

    def __get_running_jobs(self) -> List[Job]:
        j = self.__get_jobs_by_states('RUNNING')
        return self.__build_jobs_from_json(j)

    def __get_arrived_jobs(self) -> List[Job]:
        j = self.__get_jobs_by_states('NEW,NEW_SAVING,SUBMITTED,ACCEPTED')
        return self.__build_jobs_from_json(j)

    def __get_completed_jobs(self) -> List[Job]:
        j = self.__get_jobs_by_states('FINISHED,FAILED,KILLED')
        return self.__build_jobs_from_json(j)

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

        return constraint


def get_json(url: str) -> Dict[str, object]:
    r = requests.get(url)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    j = json.loads(r.text)
    return j
