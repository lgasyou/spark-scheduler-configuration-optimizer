import argparse

from optimizer.environment.abstractenv import AbstractEnv
from optimizer.environment.simulation.simulationtrainingcommunicator import SimulationTrainingCommunicator


class TrainingEnv(AbstractEnv):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def reset(self):
        self.reset_buffer()
        self.communicator.reset()

    def _communicator(self, args: argparse.Namespace):
        return SimulationTrainingCommunicator(args.simulation_host)
