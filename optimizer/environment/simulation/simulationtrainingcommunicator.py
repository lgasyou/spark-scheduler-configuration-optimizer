import os
import socket
import json
import logging
import torch
import threading
from typing import Optional

from optimizer.environment.clustercommunication.schedulerstrategy import SchedulerStrategyFactory
from optimizer.environment.clustercommunication.ievaluationcommunicator import IEvaluationCommunicator
from optimizer.environment.workloadgenerating.workloadgenerator import WorkloadGenerator
from optimizer.hyperparameters import STATE_SHAPE


class SimulationTrainingCommunicator(IEvaluationCommunicator):

    def __init__(self, simulation_host: str):
        self.HOST = simulation_host
        self.workload_generator = WorkloadGenerator()
        scheduler_type = self.get_scheduler_type()
        self.scheduler_strategy = SchedulerStrategyFactory.create(
            scheduler_type, '', '')
        self.action_set = self.scheduler_strategy.action_set

        self.state_tensor = None
        self.done_flag: bool = True

    # TODO: We must decide when is done.
    def is_done(self) -> bool:
        return False

    def act(self, action_index: int) -> float:
        s = socket.socket()
        s.connect((self.HOST, 55532))
        s.send(json.dumps({
            'queues': [
                {
                    'name': 'queueA',
                    'capacity': '25',
                    'maxCapacity': '80'
                },
                {
                    'name': 'queueB',
                    'capacity': '75',
                    'maxCapacity': '80'
                }
            ]
        }).encode('utf-8') + b'\n')
        state_json = json.loads(s.recv(10240))
        s.close()
        logging.info(state_json)
        return self.get_reward(state_json)

    def get_reward(self, state) -> float:
        return 0

    def get_state_tensor(self):
        return torch.zeros(STATE_SHAPE)

    def reset(self):
        workloads = self.generate_train_set()
        s = socket.socket()
        logging.info('%s:%d' % (self.HOST, 55533))
        s.connect((self.HOST, 55533))
        s.send(json.dumps({
            'interval': '30000',
            'numContainer': 10,
            'workloads': workloads['workloads'],
            'queues': [
                {
                    'name': 'queueA',
                    'capacity': '25',
                    'maxCapacity': '80'
                },
                {
                    'name': 'queueB',
                    'capacity': '75',
                    'maxCapacity': '80'
                }
            ]
        }).encode('utf-8') + b'\n')
        response = json.loads(s.recv(10240))
        s.close()
        state = response['state']
        self.done_flag = response['done']

    def generate_train_set(self) -> dict:
        return self.workload_generator.generate_randomly(batch_size=18, queue_partial=True)

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"
