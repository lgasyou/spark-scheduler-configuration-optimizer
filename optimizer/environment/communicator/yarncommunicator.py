from .abstractyarncommunicator import AbstractYarnCommunicator


class YarnCommunicator(AbstractYarnCommunicator):

    def __init__(self, api_url: str, hadoop_home: str):
        super().__init__(api_url, hadoop_home)

    def is_done(self) -> bool:
        return False
