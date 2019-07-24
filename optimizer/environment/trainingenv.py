import argparse

from optimizer.environment.abstractenv import AbstractEnv
from optimizer.environment.clustercommunication.trainingcommunicator import TrainingCommunicator


class TrainingEnv(AbstractEnv):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def reset(self):
        self.reset_buffer()
        self.communicator.reset()

    def _communicator(self, args: argparse.Namespace):
        return TrainingCommunicator(args.resource_manager_host, args.spark_history_server_host,
                                    args.hadoop_home, args.spark_home, args.java_home)
