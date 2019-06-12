import argparse

from .abstractenv import AbstractEnv
from .communicator import YarnCommunicator


class Env(AbstractEnv):

    def _communicator(self, args: argparse.Namespace):
        return YarnCommunicator(args.rm_host, args.hadoop_home)
