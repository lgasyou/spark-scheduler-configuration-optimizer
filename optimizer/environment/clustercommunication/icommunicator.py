import torch


class ICommunicator(object):
    """
    The interface of Communicator.
    """

    def get_state_tensor(self) -> torch.Tensor:
        pass

    def act(self, action) -> float:
        pass

    def is_done(self) -> bool:
        pass
