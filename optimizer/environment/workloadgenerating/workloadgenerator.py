import logging
import random
import json

from optimizer import hyperparameters
from optimizer.environment.workloadgenerating.facebookworkloadsampler import FacebookWorkloadSampler


class WorkloadGenerator(object):

    WORKLOAD_TYPES = ['bayes', 'FPGrowth', 'kmeans', 'lda', 'linear', 'svm']
    QUEUES = hyperparameters.QUEUES['names']
    DATA_SIZES = [str(i) for i in range(1, 9)]

    SAVE_FILENAME = './data/testset/workloads.json'

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sampler = FacebookWorkloadSampler(0, '../../../data/facebooksamples')

    def generate_randomly(self, batch_size: int = None) -> dict:
        items = []
        batch_size = batch_size or random.randint(5, 18)
        for _ in range(batch_size):
            interval, data_size = self.sampler.sample()
            item = {
                "name": random.choice(self.WORKLOAD_TYPES),
                "interval": interval,
                "queue": random.choice(self.QUEUES),
                "dataSize": data_size
            }
            items.append(item)
        self.logger.info('Generated workloads with batch size %d.' % batch_size)

        random.shuffle(items)
        return {'workloads': items}

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

    def load_evaluation_workloads(self, filename: str = None) -> dict:
        filename = filename or self.SAVE_FILENAME
        with open(filename, 'r') as f:
            workloads = json.load(f)
            self.logger.info('Workloads %s loaded.' % filename)
            return workloads

    def save_evaluation_workloads(self, workloads: dict):
        with open(self.SAVE_FILENAME, 'w') as f:
            json.dump(workloads, f)
            self.logger.info('Workloads %s saved.' % self.SAVE_FILENAME)
