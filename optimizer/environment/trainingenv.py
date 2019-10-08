import argparse

from optimizer.environment.abstractenv import AbstractEnv
from optimizer.environment.clustercommunication.trainingcommunicator import TrainingCommunicator
from optimizer.environment.simulation.simulationtrainingcommunicator import SimulationTrainingCommunicator


class TrainingEnv(AbstractEnv):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def reset(self):
        self.reset_buffer()
        self.communicator.reset()

    def _communicator(self, args: argparse.Namespace):
        if self.simulating:
            return SimulationTrainingCommunicator(args)
        return TrainingCommunicator(args)
