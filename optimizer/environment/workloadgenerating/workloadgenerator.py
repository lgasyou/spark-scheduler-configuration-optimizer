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

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sampler = FacebookWorkloadSampler()

    def generate_randomly(self, batch_size: int = None, queue_partial: bool = False) -> dict:
        items = []
        batch_size = batch_size or random.randint(5, 18)
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
        self.logger.info('Generated workloads with batch size %d.' % batch_size)

        random.shuffle(items)
        return {'workloads': items}

    def _generate_queue_by_rule(self, current_index: int, total_num: int, queue_partial: bool):
        if queue_partial:
            part_size = total_num // 3
            if current_index < part_size:
                queue = randomutil.choice_p(self.QUEUES, [2, 1])
            elif current_index < 2 * part_size:
                queue = random.choice(self.QUEUES)
            else:
                queue = randomutil.choice_p(self.QUEUES, [1, 2])
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

    def load_evaluation_workloads(self,  first_n: int = -1, filename: str = None) -> dict:
        filename = filename or self.SAVE_FILENAME
        with open(filename, 'r') as f:
            workloads = json.load(f)
            if first_n > 0:
                workloads = {'workloads': workloads['workloads'][:first_n]}
            self.logger.info('Workloads %s loaded with %d items.' % (filename, len(workloads['workloads'])))
            return workloads

    def save_evaluation_workloads(self, workloads: dict):
        with open(self.SAVE_FILENAME, 'w') as f:
            json.dump(workloads, f)
            self.logger.info('Workloads %s saved.' % self.SAVE_FILENAME)
