import random

from optimizer import hyperparameters


class SparkWorkloadRandomGenerator(object):

    WORKLOAD_TYPES = ['bayes', 'FPGrowth', 'kmeans', 'lda', 'linear', 'svm']
    QUEUES = hyperparameters.QUEUES['names']
    DATA_SIZES = [str(i) for i in range(1, 9)]

    def generate(self) -> dict:
        items = []
        workloads_size = random.randint(5, 18)
        for _ in range(workloads_size):
            item = {
                "name": random.choice(self.WORKLOAD_TYPES),
                "interval": random.randrange(3, 10),
                "queue": random.choice(self.QUEUES),
                "dataSize": random.choice(self.DATA_SIZES)
            }
            items.append(item)

        random.shuffle(items)
        return {'workloads': items}
