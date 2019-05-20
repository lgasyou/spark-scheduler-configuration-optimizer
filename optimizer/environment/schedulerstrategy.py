import os
from typing import Tuple

from .yarnmodel import *
from ..util import filecopy
from ..util.jsongetting import get_json
from ..util.xmlmodifier import XmlModifier


class FairSchedulerStrategy(object):

    def __init__(self, rm_host, hadoop_etc, action_set):
        self.rm_host = rm_host
        self.hadoop_etc = hadoop_etc
        self.action_set = action_set

    # TODO: Finish this
    def override_configuration(self, action_index: int):
        dest = os.path.join(self.hadoop_etc, 'fair-scheduler.xml')
        xml_modifier = XmlModifier('./data/fair-scheduler-template.xml', dest)

        action = self.action_set[action_index]
        a, b = action.queue_a_weight, action.queue_b_weight

        print("overriding configuration...")
        # xml_modifier.modify('yarn.scheduler.capacity.root.queueA.capacity', a)
        # xml_modifier.modify('yarn.scheduler.capacity.root.queueB.capacity', b)
        # xml_modifier.save()

    def copy_conf_file(self):
        filecopy.file_copy('./data/fair-scheduler.xml', self.hadoop_etc + '/fair-scheduler.xml')

    def get_queue_constraints(self, constraint: Constraint):
        url = self.rm_host + 'ws/v1/cluster/scheduler'
        conf = get_json(url)
        with open('./results/json.json', 'w') as f:
            f.write(str(conf))

        queues = conf['scheduler']['schedulerInfo']['rootQueue']['childQueues']
        for q in queues:
            capacity = q['capacity']
            max_capacity = q['maxCapacity']
            name = q['queueName']
            queue_c = QueueConstraint(name, capacity, max_capacity)
            constraint.add_queue_c(queue_c)


class CapacitySchedulerStrategy(object):

    def __init__(self, rm_host, hadoop_etc, action_set):
        self.rm_host = rm_host
        self.hadoop_etc = hadoop_etc
        self.action_set = action_set

    def override_configuration(self, action_index: int):
        """
        Override capacity-scheduler.xml or fair-scheduler-template.xml
        to change the capacity or weight of queues.
        """
        dest = os.path.join(self.hadoop_etc, 'capacity-scheduler.xml')
        xml_modifier = XmlModifier('./data/capacity-scheduler-template.xml', dest)

        a, b = self._get_capacities_by_action_index(action_index)

        xml_modifier.modify('yarn.scheduler.capacity.root.queueA.capacity', a)
        xml_modifier.modify('yarn.scheduler.capacity.root.queueB.capacity', b)
        xml_modifier.modify('yarn.scheduler.capacity.root.queueA.maximum-capacity', a)
        xml_modifier.modify('yarn.scheduler.capacity.root.queueB.maximum-capacity', b)

        xml_modifier.save()

    def copy_conf_file(self):
        filecopy.file_copy('./data/capacity-scheduler.xml', self.hadoop_etc + '/capacity-scheduler.xml')

    def _get_capacities_by_action_index(self, action_index: int) -> Tuple[float, float]:
        action = self.action_set[action_index]
        weight_a = action.queue_a_weight
        weight_b = action.queue_b_weight
        total_weight = weight_a + weight_b
        a = 100 * weight_a / total_weight
        b = 100 * weight_b / total_weight
        return a, b

    def get_queue_constraints(self, constraint: Constraint):
        url = self.rm_host + 'ws/v1/cluster/scheduler'
        conf = get_json(url)

        queues = conf['scheduler']['schedulerInfo']['queues']['queue']
        for q in queues:
            capacity = q['capacity']
            max_capacity = q['maxCapacity']
            name = q['queueName']
            queue_c = QueueConstraint(name, capacity, max_capacity)
            constraint.add_queue_c(queue_c)
