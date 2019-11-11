import torch

from optimizer.environment.stateobtaining.yarnmodel import State
from optimizer.hyperparameters import STATE_SHAPE, QUEUES


class StateProcessingHelper(object):

    STATE_HEIGHT = STATE_SHAPE[0]
    STATE_WIDTH = STATE_SHAPE[1]

    MAX_REQUEST_CONTAINER_COUNT = 6
    MAX_PREDICTED_DELAY = 180000000
    NUM_APP = (STATE_HEIGHT - 1) * 5
    NUM_APP_PROPERTY = 4
    NUM_QUEUE = len(QUEUES['names'])

    MAX_CAPACITY = 100

    @staticmethod
    def normalize_state_tensor(state: torch.Tensor):
        return state

    @staticmethod
    def to_tensor(state: State):
        state_list = []
        for rj in state.running_apps[:StateProcessingHelper.NUM_APP]:
            state_list.append([rj.request_container_count / StateProcessingHelper.MAX_REQUEST_CONTAINER_COUNT,
                               rj.predicted_delay / StateProcessingHelper.MAX_PREDICTED_DELAY,
                               rj.converted_location / StateProcessingHelper.NUM_QUEUE, 1])

        remain_space = StateProcessingHelper.NUM_APP - len(state_list)
        for wj in state.waiting_apps[:remain_space]:
            state_list.append([wj.request_container_count / StateProcessingHelper.MAX_REQUEST_CONTAINER_COUNT,
                               wj.predicted_delay / StateProcessingHelper.MAX_PREDICTED_DELAY,
                               wj.converted_location / StateProcessingHelper.NUM_QUEUE, 0])

        for _ in range(StateProcessingHelper.NUM_APP - len(state_list)):
            state_list.append([0] * StateProcessingHelper.NUM_APP_PROPERTY)

        queues = []
        for queue in state.constraints:
            queues.extend([queue.capacity / StateProcessingHelper.MAX_CAPACITY,
                           queue.max_capacity / StateProcessingHelper.MAX_CAPACITY])
        state_list.append(queues)

        for _ in range(4):
            state_list.append([0] * 4)

        return torch.tensor(state_list, dtype=torch.float32).reshape((20, 20))
