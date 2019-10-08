import argparse

from optimizer.environment import AbstractEnv
from optimizer.environment.clustercommunication.evaluationcommunicator import EvaluationCommunicator
from optimizer.environment.simulation.simulationevaluationcommunicator import SimulationEvaluationCommunicator


class EvaluationEnv(AbstractEnv):
    """
    High level environment implementation.
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.training = True  # Consistent with model training mode

    def reset(self):
        self.reset_buffer()
        self.communicator.reset()

    def get_total_time_cost(self) -> tuple:
        return self.communicator.get_total_time_cost()

    # Uses loss of life as terminal signal
    def train(self) -> None:
        self.training = True

    # Uses standard terminal signal
    def eval(self) -> None:
        self.training = False

    def close(self) -> None:
        self.communicator.close()

    def _communicator(self, args: argparse.Namespace):
        if self.simulating:
            return SimulationEvaluationCommunicator(args)
        return EvaluationCommunicator(args)
