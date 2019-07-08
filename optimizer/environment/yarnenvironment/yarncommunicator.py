from .abstractyarncommunicator import AbstractYarnCommunicator


class YarnCommunicator(AbstractYarnCommunicator):

    def __init__(self, rm_api_url: str, timeline_api_url: str, hadoop_home: str):
        super().__init__(rm_api_url, timeline_api_url, hadoop_home)

    def is_done(self) -> bool:
        return False

    def get_scheduler_type(self) -> str:
        return "CapacityScheduler"
