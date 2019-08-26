import json
import logging
import random

from optimizer import hyperparameters
from optimizer.environment.workloadgenerating.facebookworkloadsampler import FacebookWorkloadSampler
from optimizer.util import randomutil


class WorkloadGenerator(object):

    WORKLOAD_TYPES = ['bayes', 'FPGrowth', 'kmeans', 'lda', 'linear', 'svm']
    QUEUES = hyperparameters.QUEUES['names']
    DATA_SIZES = [str(i) for i in range(1, 9)]

    SAVE_FILENAME = './data/testset/workloads.json'

    def __init__(self, sample_index: int = 0):
        self.sampler = self.get_sampler_by_index(sample_index)

    @staticmethod
    def get_sampler_by_index(sample_index: int):
        return FacebookWorkloadSampler(sample_index)

    def reset_sampler(self, sample_index: int):
        self.sampler = FacebookWorkloadSampler(sample_index)

    def generate_randomly(self, batch_size: int = None, queue_partial: bool = False) -> dict:
        items = []
        batch_size = batch_size or random.randint(120, 210)
        for i in range(batch_size):
            interval, data_size = self.sampler.sample()
            queue = self._generate_queue_by_rule(i, batch_size, queue_partial)
            item = {
                "name": random.choice(self.WORKLOAD_TYPES),
                "interval": interval,
                "queue": queue,
                "dataSize": data_size
            }
            items.append(item)
        logging.info('Generated workloads with batch size %d.' % batch_size)

        return {'workloads': items}

    def _generate_queue_by_rule(self, current_index: int, total_num: int, queue_partial: bool):
        if queue_partial:
            part_size = total_num // 3
            if current_index < part_size:
                queue = randomutil.choice_p(self.QUEUES, [3, 1])
            elif current_index < 2 * part_size:
                queue = random.choice(self.QUEUES)
            else:
                queue = randomutil.choice_p(self.QUEUES, [1, 3])
        else:
            queue = random.choice(self.QUEUES)
        return queue

    def generate_sequentially(self):
        all_samples = self.sampler.get_all_samples()
        items = []
        for interval, data_size in all_samples:
            item = {
                "name": random.choice(self.WORKLOAD_TYPES),
                "interval": interval,
                "queue": random.choice(self.QUEUES),
                "dataSize": data_size
            }
            items.append(item)

        return {'workloads': items}

    @staticmethod
    def load_workloads(first_n: int = -1, filename: str = None) -> dict:
        filename = filename or WorkloadGenerator.SAVE_FILENAME
        with open(filename, 'r') as f:
            workloads = json.load(f)
            if first_n > 0:
                workloads['workloads'] = workloads['workloads'][:first_n]
            logging.info('Workloads %s loaded with %d items.' % (filename, len(workloads['workloads'])))
            return workloads

    @staticmethod
    def save_workloads(workloads: dict, filename: str = None):
        filename = filename or WorkloadGenerator.SAVE_FILENAME
        with open(filename, 'w') as f:
            json.dump(workloads, f)
            logging.info('Workloads %s saved.' % filename)
