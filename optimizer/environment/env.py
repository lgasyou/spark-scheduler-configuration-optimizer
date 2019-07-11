import argparse

from .abstractenv import AbstractEnv
from optimizer.environment.spark.sparkcommunicator import SparkCommunicator


class Env(AbstractEnv):

    def _communicator(self, args: argparse.Namespace):
        return SparkCommunicator(args.rm_host, args.spark_history_server_host,
                                 args.hadoop_home, args.spark_home, args.java_home)
