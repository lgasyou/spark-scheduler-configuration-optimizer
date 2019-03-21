import json
from typing import Dict, List
from xml.etree import ElementTree

import torch

from .communicator import Communicator


class Job(object):
    def __init__(self, submit_time, tasks):
        self.submit_time: int = submit_time
        self.tasks: List[Task] = tasks


class Task(object):
    def __init__(self, platform, start_time, end_time, priority):
        self.platform: str = platform
        self.start_time: int = start_time
        self.end_time: int = end_time
        self.priority: int = priority


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
    def __init__(self):
        self.jobs: List[Job] = []
        self.resources: List[Resource] = []
        self.constraint: Constraint = Constraint()


class Action(object):
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


class YarnSchedulerCommunicator(Communicator):
    """
    Manages communications with YARN cluster scheduler.
    After changing the capacity of queues, command `yarn rmadmin -refreshQueues` may be useful.
    """

    def __init__(self, hadoop_home: str):
        self.hadoop_etc = hadoop_home + '/etc'
        conf_dict = self.__read_conf()
        self.constraints = self.__parse_constraints(conf_dict)
        self.resources = self.__parse_resources(conf_dict)

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
        pass

    def get_state_tensor(self) -> torch.Tensor:
        """
        Get state of YARN which is trimmed.
        Which is defined as the ϕ(s) function defined in document.
        """
        raw_state = self.get_state()
        return torch.Tensor()

    def save_conf(self) -> None:
        """
        Save self.conf as sls-runner.xml and [capacity|fair]-scheduler.xml
        """
        pass

    def is_done(self) -> bool:
        """
        Test if all jobs are done.
        """
        pass

    def close(self) -> None:
        """
        Call save_conf function and close the communication.
        """
        self.save_conf()

    def reset(self) -> None:
        """
        Resets the env in order to run this program again.
        """
        pass

    def __read_conf(self) -> Dict[str, object]:
        """
        Read YARN scheduler configuration.
        Read sls-runner.xml first, then sls-nodes.json,
        then read capacity-scheduler.xml or fair-scheduler.xml according to sls-runner conf.
        :return: A dictionary contains configuration we get.
        """
        yarn_site_conf = self.__read_yarn_site_xml()
        sls_runner_conf = self.__read_sls_runner_xml()
        sls_nodes_conf = self.__read_sls_nodes_json()
        if 'FairScheduler' in yarn_site_conf['yarn.resourcemanager.scheduler.class']:
            scheduler_conf = self.__read_fair_scheduler_xml()
        else:
            scheduler_conf = self.__read_capacity_scheduler_xml()
        return {
            'yarn-site': yarn_site_conf,
            'sls-runner': sls_runner_conf,
            'sls-nodes': sls_nodes_conf,
            'scheduler': scheduler_conf
        }

    @staticmethod
    def __parse_resources(conf_dict: Dict[str, object]) -> List[Resource]:
        preferences = conf_dict['sls-runner']
        memory = int(preferences['yarn.sls.nm.memory.mb']) / 1024
        cpu = preferences['yarn.sls.nm.vcores']

        nodes = conf_dict['sls-nodes']['nodes']
        resources = []
        for node in nodes:
            r = Resource(node['node'], cpu, int(memory))
            resources.append(r)

        return resources

    @staticmethod
    def __parse_constraints(conf_dict: Dict[str, object]):
        scheduler_conf: Dict[str, object] = conf_dict['scheduler']
        queue_names = str(scheduler_conf['yarn.scheduler.capacity.root.queues']).split(',')
        constraint = Constraint()
        for name in queue_names:
            c = QueueConstraint(name)
            for key in scheduler_conf.keys():
                if name in key:
                    item = key.split('.')[-1]
                    if item == 'capacity':
                        c.capacity = scheduler_conf[key]
                    elif item == 'maximum-capacity':
                        c.max_capacity = scheduler_conf[key]
            constraint.add_queue_c(c)

        return constraint

    def __read_yarn_site_xml(self) -> Dict[str, object]:
        yarn_site_xml = self.hadoop_etc + '/yarn-site.xml'
        root = ElementTree.parse(yarn_site_xml).getroot()
        conf = {}
        for item in root.findall('property'):
            name = item.find('name').text
            value = item.find('value').text
            conf[name] = value
        return conf

    def __read_sls_runner_xml(self) -> Dict[str, object]:
        sls_runner_xml = self.hadoop_etc + '/sls-runner.xml'
        root = ElementTree.parse(sls_runner_xml).getroot()
        conf = {}
        for item in root.findall('property'):
            name = item.find('name').text
            value = item.find('value').text
            conf[name] = value
        return conf

    def __read_sls_nodes_json(self) -> Dict[str, object]:
        with open(self.hadoop_etc + '/sls-nodes.json') as f:
            return json.load(f)

    def __read_fair_scheduler_xml(self) -> Dict[str, object]:
        fair_scheduler_xml = self.hadoop_etc + '/fair-scheduler.xml'
        root = ElementTree.parse(fair_scheduler_xml).getroot()
        conf = {}
        for queue in root.findall('queue'):
            name = queue.attrib['name']
            weight = queue.find('weight').text
            conf[name] = weight
        return conf

    def __read_capacity_scheduler_xml(self) -> Dict[str, object]:
        capacity_scheduler_xml = self.hadoop_etc + '/capacity-scheduler.xml'
        root = ElementTree.parse(capacity_scheduler_xml).getroot()
        conf = {}
        for item in root.findall('property'):
            name = item.find('name').text
            value = item.find('value').text
            conf[name] = value
        return conf
