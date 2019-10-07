import argparse
import socket
import json
import logging
import torch

from optimizer.environment.clustercommunication.schedulerstrategy import SchedulerStrategyFactory
from optimizer.environment.clustercommunication.ievaluationcommunicator import IEvaluationCommunicator
from optimizer.environment.workloadgenerating.workloadgenerator import WorkloadGenerator
from optimizer.hyperparameters import QUEUES


class SimulationEvaluationCommunicator(IEvaluationCommunicator):

    def __init__(self, args: argparse.Namespace):
        self.HOST = args.simulation_host
        self.workload_generator = WorkloadGenerator()
        self.WORKLOADS = self.workload_generator.generate_randomly(240, queue_partial=True)
        self.workload_generator.save_workloads(self.WORKLOADS)
        scheduler_type = self.get_scheduler_type()
        self.scheduler_strategy = SchedulerStrategyFactory.create(
            scheduler_type, '', '')
        self.action_set = self.scheduler_strategy.action_set

        self.state = None
        self.done = True

    def is_done(self) -> bool:
        return self.done

    def act(self, action_index: int) -> float:
        s = socket.socket()
        s.connect((self.HOST, 55532))
        s.send(json.dumps({'queues': self.build_queue_json(action_index)}).encode('utf-8') + b'\n')
        response = json.loads(s.recv(102400, socket.MSG_WAITALL))
        s.close()
        self.state = response['state']
        self.done = response['done']
        logging.info(self.state)
        return self.get_reward(self.state)

    def get_reward(self, state) -> float:
        return -1

    def get_state_tensor(self):
        state = []
        JOB_COUNT = 50
        for rj in self.state['runJob'][:JOB_COUNT]:
            state.append([rj['container'], rj['worktime'], queue_name_to_index(rj['queue']), 1])

        remain_space = JOB_COUNT - len(state)
        for wj in self.state['waitJob'][:remain_space]:
            state.append([wj['container'], wj['worktime'], queue_name_to_index(wj['queue']), 0])

        for _ in range(JOB_COUNT - len(state)):
            state.append([0, 0, 0, 0])

        queues = []
        for queue in self.state['stricts']:
            queues.extend([queue['common'], queue['max']])
        state.append(queues)

        return torch.tensor(state, dtype=torch.float32)

    def reset(self):
        workloads = self.WORKLOADS
        s = socket.socket()
        s.connect((self.HOST, 55533))
        s.send(json.dumps({
            'interval': '1000',
            'numContainer': 10,
            'workloads': workloads['workloads'],
            'queues': self.build_queue_json(0)
        }).encode('utf-8') + b'\n')
        self.state = json.loads(s.recv(102400, socket.MSG_WAITALL))['state']
        s.close()

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"

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

    def get_total_time_cost(self):
        s = socket.socket()
        s.connect((self.HOST, 55534))
        s.send(b'\n')
        time_cost = json.loads(s.recv(102400, socket.MSG_WAITALL))
        s.close()
        return time_cost['timeCost']


def queue_name_to_index(queue_name: str) -> int:
    return QUEUES['names'].index(queue_name)
