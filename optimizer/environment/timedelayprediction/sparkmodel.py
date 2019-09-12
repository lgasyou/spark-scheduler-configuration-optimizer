from __future__ import annotations

import dataclasses
from typing import List


@dataclasses.dataclass
class Application(object):
    application_id: str
    jobs: List[Job]
    executors: List[Executor]
    input_bytes: int


@dataclasses.dataclass
class Job(object):
    job_id: int
    name: str
    stages: List[Stage]


@dataclasses.dataclass
class Stage(object):
    stage_id: int
    num_tasks: int
    input_bytes: int
    name: str
    tasks: List[Task]


@dataclasses.dataclass
class Task(object):
    task_id: int
    launch_time: int
    duration: int
    host: str
    input_bytes: int


@dataclasses.dataclass
class Executor(object):
    executor_id: int
    is_active: bool
    start_time: int
