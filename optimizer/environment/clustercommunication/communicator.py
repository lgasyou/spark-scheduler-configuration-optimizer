import argparse

from optimizer.environment.clustercommunication.abstractcommunicator import AbstractCommunicator


class Communicator(AbstractCommunicator):

    def __init__(self, args: argparse.Namespace):
        rm_host = args.rm_host
        spark_history_server_host = args.spark_history_server_host
        hadoop_home = args.hadoop_home
        super().__init__(rm_host, spark_history_server_host, hadoop_home)

    def is_done(self) -> bool:
        return False

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"
