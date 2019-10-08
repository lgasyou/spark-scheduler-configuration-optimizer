import logging
import torch

from optimizer.environment.clustercommunication.ievaluationcommunicator import IEvaluationCommunicator
from optimizer.environment.workloadgenerating.workloadgenerator import WorkloadGenerator
from optimizer.environment.clustercommunication.schedulerstrategy import SchedulerStrategyFactory
from optimizer.util import socketutil
from optimizer.hyperparameters import QUEUES


class AbstractSimulationCommunicator(IEvaluationCommunicator):

    JOB_COUNT = 50

    def __init__(self, args):
        self.HOST = args.simulation_host
        self.workload_generator = WorkloadGenerator()
        scheduler_type = self.get_scheduler_type()
        self.scheduler_strategy = SchedulerStrategyFactory.create(
            scheduler_type, '', '')
        self.action_set = self.scheduler_strategy.action_set

        self.state = None
        self.done = True

    def act(self, action_index: int) -> float:
        response = socketutil.ping_pong(self.HOST, 55532, {'queues': self.build_queue_json(action_index)})
        self.state = response['state']
        self.done = response['done']
        logging.info(self.state)
        return self.get_reward(self.state)

    def get_state_tensor(self):
        state = []
        self.JOB_COUNT = 50
        for rj in self.state['runJob'][:self.JOB_COUNT]:
            state.append([rj['container'], rj['worktime'], queue_name_to_index(rj['queue']), 1])

        remain_space = self.JOB_COUNT - len(state)
        for wj in self.state['waitJob'][:remain_space]:
            state.append([wj['container'], wj['worktime'], queue_name_to_index(wj['queue']), 0])

        for _ in range(self.JOB_COUNT - len(state)):
            state.append([0, 0, 0, 0])

        queues = []
        for queue in self.state['stricts']:
            queues.extend([queue['common'], queue['max']])
        state.append(queues)

        return torch.tensor(state, dtype=torch.float32)

    def get_reward(self, state) -> float:
        return -1

    def _submit_workloads_and_get_state(self, workloads):
        self.state = socketutil.ping_pong(self.HOST, 55533, {
            'interval': '1000',
            'numContainer': 10,
            'workloads': workloads['workloads'],
            'queues': self.build_queue_json(0)
        })['state']

    def build_queue_json(self, action_index):
        queues = []
        for name, action in self.action_set[action_index].items():
            queue = {
                'name': name,
                'capacity': action[0],
                'maxCapacity': action[1]
            }
            queues.append(queue)
        return queues


def queue_name_to_index(queue_name: str) -> int:
    return QUEUES['names'].index(queue_name)
