from optimizer.environment.abstractcommunicator import AbstractCommunicator


class SparkCommunicator(AbstractCommunicator):

    def __init__(self, rm_host: str, spark_history_server_host: str, hadoop_home: str):
        super().__init__(rm_host, spark_history_server_host, hadoop_home)

    def is_done(self) -> bool:
        return False

    def get_scheduler_type(self) -> str:
        return "capacityScheduler"
