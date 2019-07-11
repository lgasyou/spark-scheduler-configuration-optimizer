import abc
import os
import subprocess
from typing import Optional

import torch

from .actionparser import ActionParser
from .iresetablecommunicator import ICommunicator
from optimizer.environment.yarn.schedulerstrategy import SchedulerStrategyFactory
from optimizer.environment.yarn.yarnmodel import *
from optimizer.environment.yarn.statebuilder import StateBuilder


class AbstractCommunicator(ICommunicator):
    """
    Uses RESTFul API to communicate with YARN cluster scheduler.
    """

    def __init__(self, rm_api_url: str, spark_history_server_api_url: str, hadoop_home: str):
        self.hadoop_home = hadoop_home
        self.hadoop_etc = hadoop_home + '/etc/hadoop'
        self.rm_api_url = rm_api_url
        self.spark_history_server_api_url = spark_history_server_api_url + 'api/v1/'

        self.action_set = ActionParser.parse()

        scheduler_type = self.get_scheduler_type()
        self.scheduler_strategy = SchedulerStrategyFactory.create(
            scheduler_type, rm_api_url, self.hadoop_etc, self.action_set)
        self.scheduler_strategy.copy_conf_file()

        self.state_builder = StateBuilder(self.rm_api_url, self.spark_history_server_api_url, self.scheduler_strategy)

        self.state: Optional[State] = None
        self.last_sum_time_delay: Optional[float] = None

    def act(self, action_index: int) -> float:
        """
        Apply action and see how many rewards we can get.
        :return: Reward this step gets.
        """
        self.set_and_refresh_queue_config(action_index)
        return self.get_reward()

    # TODO: Test if we should use function math.tanh to clap the value of reward.
    def get_reward(self) -> float:
        waiting_jobs = self.state.waiting_apps
        running_jobs = self.state.running_apps

        # noinspection PyTypeChecker
        sum_time_delay = sum([j.predicted_time_delay for j in (waiting_jobs + running_jobs)])

        # If we just start this program, set the reward as 0.
        if self.last_sum_time_delay is None or not self.last_sum_time_delay:
            self.last_sum_time_delay = sum_time_delay
            return 0

        reward = (self.last_sum_time_delay - sum_time_delay) / self.last_sum_time_delay
        self.last_sum_time_delay = sum_time_delay
        return reward

    def get_state(self) -> State:
        """Get raw state of YARN."""
        return self.state_builder.build()

    def get_state_tensor(self) -> torch.Tensor:
        """
        Get state of YARN which is trimmed.
        Which is defined as the ϕ(s) function defined in document.

        state: {
            waiting_jobs: [WaitingJob, WaitingJob, ...],
            running_jobs: [RunningJob, RunningJob, ...],
            resources: [Resource, Resource, ...],
            queue_constraints: [QueueConstraint, QueueConstraint, ...]
        }
        """
        self.state = self.get_state()
        return self.state_builder.build_tensor(self.state)

    def set_and_refresh_queue_config(self, action_index: int) -> None:
        """
        Use script "refresh-queues.sh" to refresh the configurations of queues.
        """
        self.scheduler_strategy.override_config(action_index)
        refresh_queues(self.hadoop_home)

    @abc.abstractmethod
    def is_done(self) -> bool:
        """Test if all jobs are done."""
        pass

    @abc.abstractmethod
    def get_scheduler_type(self) -> str:
        pass

    def override_config(self, action_index: int):
        self.scheduler_strategy.override_config(action_index)


def refresh_queues(hadoop_home: str):
    subprocess.Popen([os.path.join(os.getcwd(), 'bin', 'refresh-queues.sh'), hadoop_home])
