import argparse

from optimizer.environment.abstractenv import AbstractEnv
from optimizer.environment.clustercommunication.communicator import Communicator


class Env(AbstractEnv):

    def _communicator(self, args: argparse.Namespace):
        return Communicator(args)
