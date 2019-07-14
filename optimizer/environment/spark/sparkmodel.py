from __future__ import annotations

import dataclasses
from typing import List


@dataclasses.dataclass
class Application(object):
    application_id: str
    start_time: int
    jobs: List[Job]
    executors: List[Executor]


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


@dataclasses.dataclass
class Executor(object):
    executor_id: int
    is_active: bool
    start_time: int
    max_memory_gb: float


class SparkGlobalConf(object):
    spark_block_size_bytes: int = 256 * 1024 * 1024
