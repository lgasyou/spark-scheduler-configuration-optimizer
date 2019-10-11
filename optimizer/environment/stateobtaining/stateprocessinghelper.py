import torch

from optimizer.environment.stateobtaining.yarnmodel import State
from optimizer.hyperparameters import STATE_SHAPE, QUEUES


class StateProcessingHelper(object):

    STATE_HEIGHT = STATE_SHAPE[0]
    STATE_WIDTH = STATE_SHAPE[1]

    MAX_REQUEST_CONTAINER_COUNT = 6
    MAX_PREDICTED_DELAY = 180000000
    NUM_APP = STATE_HEIGHT - 1
    NUM_APP_PROPERTY = STATE_WIDTH
    NUM_QUEUE = len(QUEUES['names'])

    MAX_CAPACITY = 100

    @staticmethod
    def normalize_state_tensor(state: torch.Tensor):
        for i in range(StateProcessingHelper.NUM_APP):
            state[i][0] /= StateProcessingHelper.MAX_REQUEST_CONTAINER_COUNT
            state[i][1] /= StateProcessingHelper.MAX_PREDICTED_DELAY
            state[i][2] /= StateProcessingHelper.NUM_QUEUE

        state[-1] /= StateProcessingHelper.MAX_CAPACITY
        return state

    @staticmethod
    def to_tensor(state: State):
        state_list = []
        for rj in state.running_apps[:StateProcessingHelper.NUM_APP]:
            state_list.append([rj.request_container_count, rj.predicted_delay, rj.converted_location, 1])

        remain_space = StateProcessingHelper.NUM_APP - len(state_list)
        for wj in state.waiting_apps[:remain_space]:
            state_list.append([wj.request_container_count, wj.predicted_delay, wj.converted_location, 0])

        for _ in range(StateProcessingHelper.NUM_APP - len(state_list)):
            state_list.append([0] * StateProcessingHelper.NUM_APP_PROPERTY)

        queues = []
        for queue in state.constraints:
            queues.extend([queue.capacity, queue.max_capacity])
        state_list.append(queues)

        return torch.tensor(state_list, dtype=torch.float32)
