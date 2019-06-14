import copy
import os

from .yarnmodel import *
from ...util import fileutil, jsonutil
from ...util.xmlmodifier import XmlModifier


class ISchedulerStrategy(object):

    def override_config(self, action_index: int):
        """
        Override capacity-scheduler.xml or fair-scheduler-template.xml
        to change the capacity or weight of queues.
        """
        pass

    def copy_conf_file(self):
        pass

    def get_queue_constraints(self, constraint: Constraint):
        pass


class FairSchedulerStrategy(ISchedulerStrategy):

    def __init__(self, rm_host, hadoop_etc, action_set):
        self.rm_host = rm_host
        self.hadoop_etc = hadoop_etc
        self.action_set = action_set
        self.current_action = None

    def override_config(self, action_index: int):
        dest = os.path.join(self.hadoop_etc, 'fair-scheduler.xml')
        xml_modifier = XmlModifier('./data/fair-scheduler-template.xml', dest)

        print("overriding configuration...")
        self.current_action: dict = self.action_set[action_index]
        for queue_name, weight in self.current_action.items():
            xml_modifier.modify_property(queue_name, weight)
        xml_modifier.save()

    def copy_conf_file(self):
        fileutil.file_copy('./data/fair-scheduler.xml', self.hadoop_etc + '/fair-scheduler.xml')

    def get_queue_constraints(self, constraint: Constraint):
        for queue_name, weight in self.current_action.items():
            constraint.add_queue_c(QueueConstraint(queue_name, weight))


class CapacitySchedulerStrategy(ISchedulerStrategy):

    def __init__(self, rm_host, hadoop_etc, action_set):
        self.rm_host = rm_host
        self.hadoop_etc = hadoop_etc
        self.action_set = self._convert_weight_to_capacity(action_set)

    def override_config(self, action_index: int):
        dest = os.path.join(self.hadoop_etc, 'capacity-scheduler.xml')
        xml_modifier = XmlModifier('./data/capacity-scheduler-template.xml', dest)

        action: dict = self.action_set[action_index]
        for queue_name, capacity in action.items():
            xml_modifier.modify_kv_type('yarn.scheduler.capacity.root.%s.capacity' % queue_name, capacity)
            xml_modifier.modify_kv_type('yarn.scheduler.capacity.root.%s.maximum-capacity' % queue_name, capacity)

        xml_modifier.save()

    def copy_conf_file(self):
        fileutil.file_copy('./data/capacity-scheduler.xml', self.hadoop_etc + '/capacity-scheduler.xml')

    @staticmethod
    def _convert_weight_to_capacity(old_action_set: dict):
        action_set = copy.deepcopy(old_action_set)
        total_weight = sum(action_set.get(0).values())
        for index, action in action_set.items():
            for queue_name, weight in action.items():
                capacity = 100 * weight / total_weight
                action[queue_name] = capacity
        return action_set

    def get_queue_constraints(self, constraint: Constraint):
        url = self.rm_host + 'ws/v1/cluster/scheduler'
        conf = jsonutil.get_json(url)

        queues = conf['scheduler']['schedulerInfo']['queues']['queue']
        for q in queues:
            capacity = q['capacity']
            max_capacity = q['maxCapacity']
            name = q['queueName']
            queue_c = QueueConstraint(name, capacity, max_capacity)
            constraint.add_queue_c(queue_c)


class SchedulerStrategyFactory(object):

    @staticmethod
    def create(scheduler_type: str, rm_host: str, hadoop_etc: str, action_set: str) -> ISchedulerStrategy:
        if scheduler_type == "FairScheduler":
            cls = FairSchedulerStrategy
        elif scheduler_type == "CapacityScheduler":
            cls = CapacitySchedulerStrategy
        else:
            raise RuntimeError
        return cls(rm_host, hadoop_etc, action_set)
