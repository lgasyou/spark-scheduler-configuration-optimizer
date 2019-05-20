from typing import List


class Job(object):
    def __init__(self, application_id, submit_time, wait_time, priority, tasks):
        self.application_id = application_id
        self.submit_time: int = submit_time
        self.wait_time: int = wait_time
        self.priority: int = priority
        self.tasks: List[Task] = tasks

    def __eq__(self, other):
        return self.application_id == other.application_id

    # Use application_id to identity a Job
    def __hash__(self):
        return hash(self.application_id)


class Task(object):
    def __init__(self, platform, memory, cpu):
        self.platform: str = platform
        self.memory: int = memory
        self.cpu: int = cpu


class Resource(object):
    def __init__(self, platform: str, cpu: int, mem: int):
        self.node_name: str = platform   # Node Name
        self.cpu: int = cpu              # CPU Cores
        self.mem: int = mem              # Memory(GB)


class QueueConstraint(object):
    def __init__(self, name, capacity=100, max_capacity=100):
        self.name = name
        self.capacity = capacity
        self.max_capacity = max_capacity


class JobConstraint(object):
    def __init__(self, job_id, host_node_name):
        self.job_id = job_id
        self.host_node_name = host_node_name


class Constraint(object):
    def __init__(self):
        self.queue: List[QueueConstraint] = []
        self.job = []

    def add_queue_c(self, c: QueueConstraint):
        self.queue.append(c)

    def add_job_c(self, c: JobConstraint):
        self.job.append(c)


class State(object):
    def __init__(self, waiting_jobs: List[Job], running_jobs: List[Job],
                 resources: List[Resource], constraint: Constraint):
        self.awaiting_jobs = waiting_jobs
        self.running_jobs: List[Job] = running_jobs
        self.resources = resources
        self.constraint = constraint


class Action(object):
    def __init__(self, a: int, b: int):
        self.queue_a_weight = a
        self.queue_b_weight = b
