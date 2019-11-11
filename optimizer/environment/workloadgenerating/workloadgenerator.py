import json
import logging
import random

from optimizer import hyperparameters
from optimizer.environment.workloadgenerating.facebookworkloadsampler import FacebookWorkloadSampler


class WorkloadGenerator(object):

    WORKLOAD_TYPES = ['bayes', 'FPGrowth', 'kmeans', 'lda', 'linear', 'svm',
                      'rnn', 'autoencoder', 'lenet', 'resnet', 'vgg']
    QUEUES = hyperparameters.QUEUES['names']
    DATA_SIZES = [str(i) for i in range(1, 9)]

    DEFAULT_SAVE_FILENAME = 'data/testset/workloads.json'

    def __init__(self, sample_index: int = 0):
        self.sampler = self.get_sampler_by_index(sample_index)

    @staticmethod
    def get_sampler_by_index(sample_index: int):
        return FacebookWorkloadSampler(sample_index)

    def reset_sampler(self, sample_index: int):
        self.sampler = FacebookWorkloadSampler(sample_index)

    def generate_randomly(self, batch_size: int = None) -> dict:
        items = []
        batch_size = batch_size or random.randint(120, 210)
        for i in range(batch_size):
            interval, data_size, queue_idx = self.sampler.sample()
            queue = self.QUEUES[queue_idx]
            item = {
                "name": random.choice(self.WORKLOAD_TYPES),
                "interval": interval,
                "queue": queue,
                "dataSize": data_size,
                "id": i + 1
            }
            items.append(item)
        logging.info('Generated workloads with batch size %d.' % batch_size)

        print(items)
        return {'workloads': items}

    def generate_sequentially(self, first_n: int = 0):
        all_samples = self.sampler.get_all_samples()
        if first_n != 0:
            all_samples = all_samples[:first_n]
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
    def load_workloads(first_n: int = 0, filename: str = None) -> dict:
        filename = filename or WorkloadGenerator.DEFAULT_SAVE_FILENAME
        with open(filename, 'r') as f:
            workloads = json.load(f)
            if first_n != 0:
                workloads['workloads'] = workloads['workloads'][:first_n]
            logging.info('Workloads %s loaded with %d items.' % (filename, len(workloads['workloads'])))
            return workloads

    @staticmethod
    def save_workloads(workloads: dict, filename: str = None):
        filename = filename or WorkloadGenerator.DEFAULT_SAVE_FILENAME
        with open(filename, 'w') as f:
            json.dump(workloads, f)
            logging.info('Workloads %s saved.' % filename)
