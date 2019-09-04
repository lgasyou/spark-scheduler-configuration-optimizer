import torch
from typing import List

from optimizer.environment.stateobtaining.yarnmodel import State, ApplicationRequestResource
from optimizer.hyperparameters import STATE_SHAPE


class StateProcessor(object):

    MAX_NODE_MEM_GB = 512
    MAX_NODE_VCORE_NUM = 100

    MAX_CAPACITY = 100
    MAX_MAX_CAPACITY = 100
    MAX_USED_CAPACITY = 100

    MAX_ELAPSED_SECONDS = 100000
    MAX_PRIORITY = 100
    MAX_QUEUE_USAGE_PERCENTAGE = 100
    MAX_PREDICTED_DELAY_SECONDS = 100000

    MAX_REQUEST_MEM_GB = 512
    MAX_REQUEST_VCORE_NUM = 100

    def normalize_state(self, state: State):
        for r in state.resources:
            r.mem /= self.MAX_NODE_MEM_GB
            r.vcore_num /= self.MAX_NODE_VCORE_NUM

        for c in state.constraints:
            c.capacity /= self.MAX_CAPACITY
            c.max_capacity /= self.MAX_MAX_CAPACITY
            c.used_capacity /= self.MAX_USED_CAPACITY

        for ra in state.running_apps:
            ra.elapsed_time /= self.MAX_ELAPSED_SECONDS
            ra.priority /= self.MAX_PRIORITY
            ra.queue_usage_percentage /= self.MAX_QUEUE_USAGE_PERCENTAGE
            ra.predicted_delay /= self.MAX_PREDICTED_DELAY_SECONDS
            self._normalize_request_resources(ra.request_resources)

        for wa in state.waiting_apps:
            wa.elapsed_time /= self.MAX_ELAPSED_SECONDS
            wa.priority /= self.MAX_PRIORITY
            wa.predicted_delay /= self.MAX_PREDICTED_DELAY_SECONDS
            self._normalize_request_resources(wa.request_resources)

        return state

    def _normalize_request_resources(self, request_resources: List[ApplicationRequestResource]):
        for rr in request_resources:
            rr.priority /= self.MAX_PRIORITY
            rr.memory /= self.MAX_REQUEST_MEM_GB
            rr.vcore_num /= self.MAX_REQUEST_VCORE_NUM

    # TODO: Redesign this, with three channels.
    @staticmethod
    def to_tensor(normalized_state: State):
        height, width = STATE_SHAPE
        tensor = torch.zeros(height, width)

        # Line 0-74: waiting apps and their resource requests
        for i, wa in enumerate(normalized_state.waiting_apps[:75]):
            line = [wa.elapsed_time, wa.priority, wa.converted_location]
            for rr in wa.request_resources[:64]:
                line.extend([rr.priority, rr.memory, rr.vcore_num])
            line.extend([0.0] * (width - len(line)))
            tensor[i] = torch.Tensor(line)

        # Line 75-149: running apps and their resource requests
        for i, ra in enumerate(normalized_state.running_apps[:75]):
            row = i + 75
            line = [ra.elapsed_time, ra.priority, ra.converted_location,
                    ra.progress, ra.queue_usage_percentage, ra.predicted_delay]
            for rr in ra.request_resources[:65]:
                line.extend([rr.priority, rr.memory, rr.vcore_num])
            line.extend([0.0] * (width - len(line)))
            tensor[row] = torch.Tensor(line)

        # TODO: 是否需要这个粒度的数据
        # Line 150-198: resources of cluster
        row, idx = 150, 0
        for r in normalized_state.resources[:4900]:
            tensor[row][idx] = r.mem
            idx += 1
            tensor[row][idx] = r.vcore_num
            idx += 1
            if idx == width:
                row += 1
                idx = 0

        # Line 199: queue constraints
        row, queue_constraints = 199, []
        for c in normalized_state.constraints[:50]:
            queue_constraints.extend([c.converted_name, c.capacity, c.max_capacity, c.used_capacity])
        queue_constraints.extend([0.0] * (width - len(queue_constraints)))
        tensor[row] = torch.Tensor(queue_constraints)

        return tensor
