import json
import os
import subprocess
import time
from typing import Dict, List, Tuple

import requests
import torch

from .communicator import Communicator
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
        self.waiting_jobs = waiting_jobs
        self.running_jobs = running_jobs,
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

    def __init__(self, rm_host: str, hadoop_home: str, sls_jobs_json: str):
        self.hadoop_home = hadoop_home
        self.hadoop_etc = hadoop_home + '/etc'
        self.rm_host = rm_host
        self.sls_jobs_json = sls_jobs_json

        self.waiting_jobs: List[Job] = []

        self.action_set = self.get_action_set()
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

    def act_before_start(self, action_index: int):
        """
        Act before YARN starts.
        Override capacity-scheduler.xml or fair-scheduler.xml.
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
        :return: Reward this step got.
        """
        action = self.action_set[action_index]
        a = action.queue_a_weight
        b = action.queue_b_weight
        self.set_queue_weights(a, b)
        return self.get_reward()

    def get_reward(self) -> float:
        job_count = len(self.waiting_jobs)
        if job_count == 0:
            return 0

        total_wait_time = 0.0
        for j in self.waiting_jobs:
            total_wait_time += j.wait_time

        return total_wait_time / job_count

    def get_state(self) -> State:
        """
        Get raw state of YARN.
        """
        wj, rj = self.__get_jobs()
        resources = self.__get_resources()
        constraints = self.__get_constraints()
        self.waiting_jobs = wj
        return State(wj, rj, resources, constraints)

    # TODO: Transform into tensor
    def get_state_tensor(self) -> torch.Tensor:
        """
        Get state of YARN which is trimmed.
        Which is defined as the ϕ(s) function defined in document.

        state: {
            waiting_jobs: [Job, Job, ...],
            running_jobs: [Job, Job, ...],
            resources: [Resource, Resource, ...],
            constraint: {
                queue: [QueueConstraint, QueueConstraint, ...],
                job: [JobConstraint, JobConstraint, ...],
            }
        }

        Line 1: waiting_jobs
        Line 2: running_jobs
        Line 3: resources
        Line 4: queue
        Line 5: job
        """
        raw = self.get_state()

        for wj in raw.waiting_jobs:
            pass

        for rj in raw.running_jobs:
            pass

        for r in raw.resources:
            pass

        for c in raw.constraint.queue:
            pass

        for j in raw.constraint.job:
            pass

        return torch.Tensor(42, 42)

    # TODO: Bug Configuration change only supported by MutableConfScheduler.
    def set_queue_weights(self, queue_a_weight: int, queue_b_weight: int) -> None:
        """
        Set queue weights by using RESTFul API.
        """
        total_weight = queue_a_weight + queue_b_weight
        a = 100 * queue_a_weight / total_weight
        b = 100 * queue_b_weight / total_weight
        conf = """
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <sched-conf>
          <update-queue>
            <queue-name>root.queueA</queue-name>
            <params>
              <entry>
                <key>capacity</key>
                <value>{}</value>
              </entry>
            </params>
          </update-queue>
          <update-queue>
            <queue-name>root.queueB</queue-name>
            <params>
              <entry>
                <key>capacity</key>
                <value>{}</value>
              </entry>
            </params>
          </update-queue>
        </sched-conf>
        """.format(a, b)
        url = self.rm_host + 'ws/v1/cluster/scheduler-conf'

        # r = requests.put(url, conf, headers={
        #     'Context-Type': 'application/xml'
        # })
        # r.raise_for_status()

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

    def reset(self, wd=None) -> None:
        """
        Resets the env in order to run this program again.
        """
        if wd is None:
            wd = os.getcwd()

        self.close()

        sls_jobs_json = wd + '/' + self.sls_jobs_json
        cmd = [wd + '/start-sls.sh', self.hadoop_home, wd, sls_jobs_json]
        self.sls_runner = subprocess.Popen(cmd)

        # Wait until web server starts.
        time.sleep(8)

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
        for job in self.waiting_jobs:
            url = "{}ws/v1/cluster/apps/{}/appattempts".format(self.rm_host, job.application_id)
            attempt_json = get_json(url)
            attempts = attempt_json['appAttempts']['appAttempt']
            for a in attempts:
                http_address = a['nodeHttpAddress']
                if len(http_address):
                    constraint.add_job_c(JobConstraint(job.application_id, http_address))

        return constraint


def get_current_time_ms() -> int:
    return int(time.time() * 1000)


def get_json(url: str) -> Dict[str, object]:
    r = requests.get(url)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    j = json.loads(r.text)
    return j
