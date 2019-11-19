from optimizer.environment.clustercommunication.abstractcommunicator import AbstractCommunicator


class Communicator(AbstractCommunicator):

    def is_done(self) -> bool:
        return False
