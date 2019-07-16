import copy
import os

from optimizer.environment.yarn.yarnmodel import *
from optimizer.util import fileutil, jsonutil
from optimizer.util.xmlmodifier import XmlModifier


class ISchedulerStrategy(object):

    def override_config(self, action_index: int):
        """
        Override capacity-scheduler.xml or fair-scheduler-template.xml
        to change the capacity or weight of queues.
        """
        pass

    def copy_conf_file(self):
        pass

    def get_queue_constraints(self) -> List[QueueConstraint]:
        pass


class FairSchedulerStrategy(ISchedulerStrategy):

    def __init__(self, rm_host, hadoop_etc, action_set):
        self.RM_HOST = rm_host
        self.HADOOP_ETC = hadoop_etc
        self.action_set = action_set
        self.current_action = None

    def override_config(self, action_index: int):
        dest = os.path.join(self.HADOOP_ETC, 'fair-scheduler.xml')
        xml_modifier = XmlModifier('./data/fair-scheduler-template.xml', dest)

        print("overriding configuration...")
        self.current_action: dict = self.action_set[action_index]
        for queue_name, weight in self.current_action.items():
            xml_modifier.modify_property(queue_name, weight)
        xml_modifier.save()

    def copy_conf_file(self):
        fileutil.file_copy('./data/fair-scheduler.xml', self.HADOOP_ETC + '/fair-scheduler.xml')

    @DeprecationWarning
    def get_queue_constraints(self):
        return [QueueConstraint('', queue_name, weight, weight) for queue_name, weight in self.current_action.items()]


class CapacitySchedulerStrategy(ISchedulerStrategy):

    def __init__(self, rm_host, hadoop_etc, action_set):
        self.RM_HOST = rm_host
        self.HADOOP_ETC = hadoop_etc
        self.action_set = self._convert_weight_to_capacity(action_set)

    def override_config(self, action_index: int):
        dest = os.path.join(self.HADOOP_ETC, 'capacity-scheduler.xml')
        xml_modifier = XmlModifier('./data/capacity-scheduler-template.xml', dest)

        action: dict = self.action_set[action_index]
        for queue_name, capacity in action.items():
            xml_modifier.modify_kv_type('yarn.scheduler.capacity.root.%s.capacity' % queue_name, capacity)
            # TODO: edit this
            xml_modifier.modify_kv_type('yarn.scheduler.capacity.root.%s.maximum-capacity' % queue_name, 80)

        xml_modifier.save()

    def copy_conf_file(self):
        fileutil.file_copy('./data/capacity-scheduler.xml', self.HADOOP_ETC + '/capacity-scheduler.xml')

    def get_queue_constraints(self):
        url = self.RM_HOST + 'ws/v1/cluster/scheduler'
        conf = jsonutil.get_json(url)

        ret = []
        queues = conf['scheduler']['schedulerInfo']['queues']['queue']
        for q in queues:
            name = q['queueName']
            capacity = q['capacity']
            used_capacity = q['usedCapacity']
            max_capacity = q['maxCapacity']
            ret.append(QueueConstraint(name, used_capacity, capacity, max_capacity))

        return ret

    @staticmethod
    def _convert_weight_to_capacity(old_action_set: dict):
        action_set = copy.deepcopy(old_action_set)
        for index, action in action_set.items():
            total_weight = sum(action.values())
            for queue_name, weight in action.items():
                capacity = 100 * weight / total_weight
                action[queue_name] = round(capacity, 2)
        return action_set


class SchedulerStrategyFactory(object):

    STR_STRATEGY_MAP = {
        'FairScheduler': FairSchedulerStrategy,
        'CapacityScheduler': CapacitySchedulerStrategy
    }

    @staticmethod
    def create(scheduler_type: str, rm_host: str, hadoop_etc: str, action_set: dict) -> ISchedulerStrategy:
        cls = SchedulerStrategyFactory.STR_STRATEGY_MAP.get(scheduler_type)
        if cls is None:
            raise RuntimeError

        return cls(rm_host, hadoop_etc, action_set)
