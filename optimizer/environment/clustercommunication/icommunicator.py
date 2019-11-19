import torch


class ICommunicator(object):
    """
    The interface of Communicator.
    """

    def get_state_tensor(self) -> torch.Tensor:
        pass

    def act(self, action) -> float:
        """
        Apply action and see how many rewards we can get.
        :return: Reward this step gets.
        """
        pass

    def is_done(self) -> bool:
        """Test if all jobs are done."""
        pass
