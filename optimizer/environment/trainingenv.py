import argparse

from optimizer.environment.abstractenv import AbstractEnv
from optimizer.environment.clustercommunication.trainingcommunicator import TrainingCommunicator


class TrainingEnv(AbstractEnv):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def generate_pre_train_set(self) -> dict:
        return self.communicator.generate_pre_train_set()

    def start(self, workloads):
        self._reset(workloads)

    def step(self, action, retry_interval: int):
        state = self.try_get_state(retry_interval)
        reward = self.communicator.act(action)
        done = self.communicator.is_done()
        return state, reward, done

    def _communicator(self, args: argparse.Namespace):
        return TrainingCommunicator(args.resource_manager_host, args.spark_history_server_host,
                                    args.hadoop_home, args.spark_home, args.java_home)

    def _reset(self, workloads):
        self.reset_buffer()
        self.communicator.reset(workloads)
