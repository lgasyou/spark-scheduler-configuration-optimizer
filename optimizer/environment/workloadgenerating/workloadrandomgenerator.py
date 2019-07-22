import logging
import random

from optimizer import hyperparameters
from optimizer.environment.workloadgenerating.facebookworkloadsampler import FacebookWorkloadSampler


class WorkloadRandomGenerator(object):

    WORKLOAD_TYPES = ['bayes', 'FPGrowth', 'kmeans', 'lda', 'linear', 'svm']
    QUEUES = hyperparameters.QUEUES['names']
    DATA_SIZES = [str(i) for i in range(1, 9)]

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sampler = FacebookWorkloadSampler()

    def generate(self, batch_size: int = None) -> dict:
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
