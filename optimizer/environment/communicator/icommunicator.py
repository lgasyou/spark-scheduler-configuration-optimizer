import torch


class ICommunicator(object):
    """
    The interface of Communicator.
    """

    @staticmethod
    def get_action_set() -> dict:
        pass

    def get_state_tensor(self) -> torch.Tensor:
        pass

    def act(self, action) -> float:
        pass

    def is_done(self) -> bool:
        pass
