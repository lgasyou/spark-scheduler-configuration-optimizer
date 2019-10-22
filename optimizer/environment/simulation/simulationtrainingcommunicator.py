import argparse

from optimizer.environment.simulation.abstractsimulationcommunicator import AbstractSimulationCommunicator


class SimulationTrainingCommunicator(AbstractSimulationCommunicator):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def is_done(self) -> bool:
        return self.done

    def reset(self):
        workloads = self.generate_train_set()
        self._submit_workloads_and_get_state(workloads)

    def generate_train_set(self) -> dict:
        return self.workload_generator.generate_randomly(batch_size=240, queue_partial=True)

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"
