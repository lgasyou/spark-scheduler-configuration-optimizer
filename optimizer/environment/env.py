import argparse

from optimizer.environment.abstractenv import AbstractEnv
from optimizer.environment.clustercommunication.communicator import Communicator


class Env(AbstractEnv):

    def _communicator(self, args: argparse.Namespace):
        return Communicator(args.resource_manager_host, args.spark_history_server_host, args.hadoop_home)
