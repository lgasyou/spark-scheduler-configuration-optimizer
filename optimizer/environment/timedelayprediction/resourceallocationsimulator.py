from optimizer.util import timeutil


class ResourceAllocationSimulator(object):

    def __init__(self):
        self.free_resources = dict()

    def set_resources(self, resources: list):
        self.free_resources[timeutil.current_time_ms()] = resources

    # Returns allocated and unallocated number of containers.
    def allocate_once(self, time: int, queue: int, num_containers: int):
        allocated = min(self.resources_of(time, queue), num_containers)
        if time in self.free_resources:
            self.free_resources[time][queue] -= allocated
        return allocated, num_containers - allocated

    def allocate(self, queue: int, num_containers: int):
        queue, num_containers = int(queue), int(num_containers)
        ret = []
        remaining = num_containers
        for time, queues in self.free_resources.items():
            if remaining == 0:
                break

            if queues[queue] != 0:
                allocated, remaining = self.allocate_once(time, queue, remaining)
                ret.append((time, allocated))
        return ret

    def release(self, time: int, queue: int, num_containers: int):
        time, queue, num_containers = int(time), int(queue), int(num_containers)
        if time in self.free_resources:
            self.free_resources[time][queue] += num_containers
        else:
            self.free_resources[time] = [0, 0]
            self.free_resources[time][queue] = num_containers
        self.free_resources = dict(sorted(self.free_resources.items()))

    def resources_of(self, time: int, queue: int):
        if time not in self.free_resources:
            return 0
        return self.free_resources[time][queue]