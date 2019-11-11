import argparse

from optimizer.environment.simulation.abstractsimulationcommunicator import AbstractSimulationCommunicator


class SimulationTrainingCommunicator(AbstractSimulationCommunicator):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.WORKLOADS = self.generate_train_set()

    def is_done(self) -> bool:
        return self.done

    def reset(self):
        self._submit_workloads_and_get_state(self.WORKLOADS)

    def generate_train_set(self) -> dict:
        return self.workload_generator.generate_randomly(batch_size=240)

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"
