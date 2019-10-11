import logging

from optimizer.environment.clustercommunication.ievaluationcommunicator import IEvaluationCommunicator
from optimizer.environment.workloadgenerating.workloadgenerator import WorkloadGenerator
from optimizer.environment.clustercommunication.schedulerstrategy import SchedulerStrategyFactory
from optimizer.environment.stateobtaining.stateprocessinghelper import StateProcessingHelper
from optimizer.environment.simulation.simulationstatebuilder import SimulationStateBuilder
from optimizer.util import socketutil


class AbstractSimulationCommunicator(IEvaluationCommunicator):

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
        state_dict = response['state']
        self.state = SimulationStateBuilder.build(state_dict)
        self.done = response['done']
        logging.info(self.state)
        return self.get_reward(self.state)

    def get_state_tensor(self):
        state_tensor = StateProcessingHelper.to_tensor(self.state)
        return StateProcessingHelper.normalize_state_tensor(state_tensor)

    def get_reward(self, state) -> float:
        return -1

    def _submit_workloads_and_get_state(self, workloads):
        state_dict = socketutil.ping_pong(self.HOST, 55533, {
            'interval': '1000',
            'numContainer': 10,
            'workloads': workloads['workloads'],
            'queues': self.build_queue_json(0)
        })['state']
        self.state = SimulationStateBuilder.build(state_dict)

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
