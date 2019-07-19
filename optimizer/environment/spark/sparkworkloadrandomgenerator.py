import random

from optimizer import hyperparameters


class SparkWorkloadRandomGenerator(object):

    WORKLOAD_TYPES = ['bayes', 'FPGrowth', 'kmeans', 'lda', 'linear', 'SVM']
    QUEUES = hyperparameters.QUEUES['names']
    DATA_SIZES = [str(i) for i in range(1, 9)]

    def generate(self) -> dict:
        items = []
        for workload in self.WORKLOAD_TYPES:
            for _ in range(3):
                item = {
                    "name": workload,
                    "interval": random.randrange(3, 10),
                    "queue": random.choice(self.QUEUES),
                    "dataSize": random.choice(self.DATA_SIZES)
                }
                items.append(item)

        random.shuffle(items)
        return {'workloads': items}
