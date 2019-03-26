import json

import numpy as np
import torch


class GoogleTraceParser(object):
    """
    Parse the data in "data/trainingset"
    """

    def __init__(self, directory: str):
        self.dir = directory
        self.max_job_num = 100
        self.max_task_num_per_job = 1000

    def parse(self, index: int, hour: int, save_as_csv=False) -> torch.Tensor:
        filename_without_extension = "{}/{}/sls_jobs{}".format(self.dir, index, hour)
        filename = filename_without_extension + '.json'

        with open(filename, 'r') as f:
            job_lines = []
            data = np.zeros(shape=(self.max_job_num, self.max_task_num_per_job), dtype=np.int64)
            job_cnt = 0
            for line in f:
                job_lines.append(line)
                if line[0] == '}':
                    j = json.loads(''.join(job_lines))
                    job = self.__generate_line(j)
                    data[job_cnt, :] = job
                    job_cnt += 1
                    job_lines.clear()

        arr = np.array(data, dtype=np.int64)
        if save_as_csv:
            np.save(filename_without_extension, arr)

        return torch.from_numpy(arr)

    def __generate_line(self, job) -> np.ndarray:
        a = np.zeros(self.max_task_num_per_job, dtype=np.int64)
        a[0] = job['job.start.ms']
        last_index = 0
        for i, task in enumerate(job['job.tasks']):
            actual_index = i + 1
            a[actual_index] = task['container.end.ms']
            last_index = actual_index
        a[last_index + 1] = len(job['job.tasks'])
        return a
