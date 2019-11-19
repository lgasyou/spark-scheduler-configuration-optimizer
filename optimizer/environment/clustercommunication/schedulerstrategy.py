import abc
import os
from typing import Dict, Type

from optimizer.environment.stateobtaining.yarnmodel import *
from optimizer.util import fileutil, jsonutil
from optimizer.util.xmlmodifier import XmlModifier


class AbstractSchedulerStrategy(object):

    def __init__(self, args):
        self.TYPE = args.scheduler_type
        self.RM_HOST = args.resource_manager_host
        self.HADOOP_ETC = args.hadoop_etc
        self.IS_SIMULATING = args.use_simulation_env
        self.action_set = self._get_action_set()

    @abc.abstractmethod
    def override_config(self, action_index: int):
        """
        Override capacity-scheduler.xml or fair-scheduler-template.xml
        to change the capacity or weight of queues.
        """
        pass

    @abc.abstractmethod
    def copy_conf_file(self):
        pass

    @abc.abstractmethod
    def get_queue_constraints(self) -> List[QueueConstraint]:
        pass

    def _get_action_set(self):
        queue_names = QUEUES.get("names")
        raw_actions = QUEUES.get("actions")[self.TYPE]
        return self._build_actions(queue_names, raw_actions)

    def _build_actions(self, queue_names, raw_actions):
        action_space = len(raw_actions)
        actions = {}
        for i in range(action_space):
            action = self._build_action(i, queue_names, raw_actions)
            actions[i] = action

        return actions

    @abc.abstractmethod
    def _build_action(self, index: int, queue_names: list, actions: dict):
        pass


class FairSchedulerStrategy(AbstractSchedulerStrategy):

    def override_config(self, action_index: int):
        dest = os.path.join(self.HADOOP_ETC, 'fair-scheduler.xml')
        xml_modifier = XmlModifier('data/fair-scheduler-template.xml', dest)

        current_action = self.action_set[action_index]
        weights: dict = current_action['weights']
        for queue_name, weight in weights.items():
            xml_modifier.modify_property(queue_name, weight)
        xml_modifier.modify_all_values('schedulingPolicy', current_action['schedulingPolicy'])

        xml_modifier.save()

    def copy_conf_file(self):
        if not self.IS_SIMULATING:
            fileutil.file_copy('data/fair-scheduler.xml', self.HADOOP_ETC + '/fair-scheduler.xml')

    def get_queue_constraints(self):
        url = self.RM_HOST + 'ws/v1/cluster/scheduler'
        conf = jsonutil.get_json(url)

        ret = []
        root_queue = conf['scheduler']['schedulerInfo']['rootQueue']
        queues = root_queue['childQueues']['queue']
        for q in queues:
            name = q['queueName'].split('.')[-1]
            if name == 'default':
                continue

            capacity = self.calculate_percentage(q['maxResources'], q['clusterResources'])
            max_capacity = capacity
            ret.append(QueueConstraint(capacity, max_capacity))

        return ret

    @staticmethod
    def calculate_percentage(lhs: dict, rhs: dict):
        return (lhs['vCores'] / rhs['vCores'] +
                lhs['memory'] / rhs['memory']) * 50

    def _build_action(self, index: int, queue_names: list, actions: dict):
        weights, scheduling_policy = actions.get(index)
        action = {}
        for name, weight in zip(queue_names, weights):
            action[name] = weight
        return {
            'weights': action,
            'schedulingPolicy': scheduling_policy
        }


class CapacitySchedulerStrategy(AbstractSchedulerStrategy):

    def override_config(self, action_index: int):
        dest = os.path.join(self.HADOOP_ETC, 'capacity-scheduler.xml')
        xml_modifier = XmlModifier('data/capacity-scheduler-template.xml', dest)

        action: dict = self.action_set[action_index]
        for queue_name, conf in action.items():
            capacity, max_capacity = conf
            xml_modifier.modify_kv_type('yarn.scheduler.capacity.root.%s.capacity' % queue_name, capacity)
            xml_modifier.modify_kv_type('yarn.scheduler.capacity.root.%s.maximum-capacity' % queue_name, max_capacity)

        xml_modifier.save()

    def copy_conf_file(self):
        if not self.IS_SIMULATING:
            fileutil.file_copy('data/capacity-scheduler.xml', self.HADOOP_ETC + '/capacity-scheduler.xml')

    def get_queue_constraints(self):
        url = self.RM_HOST + 'ws/v1/cluster/scheduler'
        conf = jsonutil.get_json(url)

        ret = []
        queues = conf['scheduler']['schedulerInfo']['queues']['queue']
        for q in queues:
            capacity = q['capacity']
            max_capacity = q['maxCapacity']
            ret.append(QueueConstraint(capacity, max_capacity))

        return ret

    def _build_action(self, index: int, queue_names: list, actions: dict):
        capacities, max_capacities = actions.get(index)
        action = {}
        for name, c, mc in zip(queue_names, capacities, max_capacities):
            action[name] = (c, mc)
        return action


class SchedulerStrategyFactory(object):

    STR_STRATEGY_MAP: Dict[str, Type[AbstractSchedulerStrategy]] = {
        'fairScheduler': FairSchedulerStrategy,
        'capacityScheduler': CapacitySchedulerStrategy
    }

    @staticmethod
    def create(args) -> AbstractSchedulerStrategy:
        try:
            cls = SchedulerStrategyFactory.STR_STRATEGY_MAP[args.scheduler_type]
        except KeyError:
            raise RuntimeError('No strategy for %s.' % args.scheduler_type)

        return cls(args)
