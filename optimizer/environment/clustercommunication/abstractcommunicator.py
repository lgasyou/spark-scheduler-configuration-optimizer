import abc
import time
from typing import Optional

import torch

from optimizer import hyperparameters
from optimizer.environment.clustercommunication.icommunicator import ICommunicator
from optimizer.environment.clustercommunication.schedulerstrategy import SchedulerStrategyFactory
from optimizer.environment.stateobtaining.rewardcalculator import RewardCalculator
from optimizer.environment.stateobtaining.statebuilder import StateBuilder
from optimizer.environment.stateobtaining.yarnmodel import *
from optimizer.util import yarnutil


class AbstractCommunicator(ICommunicator):

    def __init__(self, rm_host: str, spark_history_server_host: str, hadoop_home: str):
        self.HADOOP_HOME = hadoop_home
        self.HADOOP_ETC = hadoop_home + '/etc/hadoop'
        self.RM_API_URL = rm_host
        self.SPARK_HISTORY_SERVER_API_URL = spark_history_server_host + 'api/v1/'

        scheduler_type = self.get_scheduler_type()
        self.scheduler_strategy = SchedulerStrategyFactory.create(
            scheduler_type, self.RM_API_URL, self.HADOOP_ETC)
        self.scheduler_strategy.copy_conf_file()
        self.action_set = self.scheduler_strategy.action_set

        self.state_builder = StateBuilder(self.RM_API_URL, self.SPARK_HISTORY_SERVER_API_URL, self.scheduler_strategy)
        self.reward_calculator = RewardCalculator()

        self.state: Optional[State] = None

    def act(self, action_index: int) -> float:
        """
        Apply action and see how many rewards we can get.
        :return: Reward this step gets.
        """
        self.set_and_refresh_queue_config(action_index)
        self._wait_for_application()
        return self.get_reward()

    def get_reward(self) -> float:
        return self.reward_calculator.get_reward(self.state)

    def get_state_tensor(self) -> torch.Tensor:
        """
        Get state which is trimmed.
        Which is defined as the ϕ(s) function defined in document.

        state: {
            waiting_jobs: [WaitingJob, WaitingJob, ...],
            running_jobs: [RunningJob, RunningJob, ...],
            resources: [Resource, Resource, ...],
            queue_constraints: [QueueConstraint, QueueConstraint, ...]
        }
        """
        self.state = self.state_builder.build()
        normalized_state = self.state_builder.normalize_state(self.state)
        return self.state_builder.build_tensor(normalized_state)

    def set_and_refresh_queue_config(self, action_index: int) -> None:
        """Use script "refresh-queues.sh" to refresh the configurations of queues."""
        self.scheduler_strategy.override_config(action_index)
        yarnutil.refresh_queues(self.HADOOP_HOME)

    @abc.abstractmethod
    def is_done(self) -> bool:
        """Test if all jobs are done."""
        pass

    @abc.abstractmethod
    def get_scheduler_type(self) -> str:
        pass

    def override_config(self, action_index: int):
        self.scheduler_strategy.override_config(action_index)

    @staticmethod
    def _wait_for_application():
        time.sleep(hyperparameters.WAIT_CONFIG_APPLY_TIME)
