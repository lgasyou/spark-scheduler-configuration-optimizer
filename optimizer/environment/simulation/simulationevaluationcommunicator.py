import argparse

from optimizer.environment.simulation.abstractsimulationcommunicator import AbstractSimulationCommunicator
from optimizer.util import socketutil


class SimulationEvaluationCommunicator(AbstractSimulationCommunicator):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.WORKLOADS = self.workload_generator.generate_randomly(240)

    def is_done(self) -> bool:
        return self.done

    def reset(self):
        self._submit_workloads_and_get_state(self.WORKLOADS)

    def get_finished_job_number(self):
        response = socketutil.ping_pong(self.HOST, 55535)
        return response['finishedJobNumber']

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"
