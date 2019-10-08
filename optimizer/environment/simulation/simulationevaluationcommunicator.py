import argparse

from optimizer.environment.simulation.abstractsimulationcommunicator import AbstractSimulationCommunicator
from optimizer.util import socketutil


class SimulationEvaluationCommunicator(AbstractSimulationCommunicator):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.WORKLOADS = self.workload_generator.generate_randomly(240, queue_partial=True)
        self.workload_generator.save_workloads(self.WORKLOADS)

    def is_done(self) -> bool:
        return self.done

    def reset(self):
        self._submit_workloads_and_get_state(self.WORKLOADS)

    def get_total_time_cost(self):
        response = socketutil.ping_pong(self.HOST, 55534)
        return response['timeCost']

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"
