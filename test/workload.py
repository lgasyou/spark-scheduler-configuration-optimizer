import json
import os
import random
import logging


def choice_p(items: list, p: list):
    h = [0] * len(p)
    h[0] = p[0]
    for i in range(1, len(p)):
        h[i] = h[i - 1] + p[i]

    p = random.randint(0, sum(p) - 1)
    for index, i in enumerate(h):
        if p < i:
            return items[index]


class FacebookWorkloadSampler(object):

    SAMPLE_FILENAMES = [
        'FB-2009_samples_24_times_1hr_0.tsv',
        'FB-2009_samples_24_times_1hr_1.tsv',
        'FB-2010_samples_24_times_1hr_0.tsv',
        'FB-2010_samples_24_times_1hr_withInputPaths_0.tsv'
    ]

    def __init__(self, sampler_index: int = 0, directory: str = '../data/facebooksamples'):
        print(self.SAMPLE_FILENAMES[sampler_index])
        filename = os.path.join(directory, self.SAMPLE_FILENAMES[sampler_index])
        with open(filename, 'r') as f:
            self.samples = f.readlines()
            self.num_samples = len(self.samples)

    def sample(self):
        while True:
            line_num = random.randint(0, self.num_samples - 1)
            sample = self.samples[line_num]
            raw_interval, data_size = self._parse_line(sample)
            interval = raw_interval if raw_interval < 100 else 100
            return interval, str(data_size)

    def get_all_samples(self):
        return [self._parse_line(line) for line in self.samples]

    @staticmethod
    def _parse_line(line: str):
        data = line.split('\t')
        interval = int(data[2])
        data_size = (len(data[3]) + 1) // 2
        return interval, data_size


class WorkloadGenerator(object):

    WORKLOAD_TYPES = ['bayes', 'FPGrowth', 'kmeans', 'lda', 'linear', 'svm']
    QUEUES = ["queueA", "queueB"]
    DATA_SIZES = [str(i) for i in range(1, 9)]

    SAVE_FILENAME = './data/testset/workloads.json'

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sampler = FacebookWorkloadSampler(3, '../data/facebooksamples')

    def generate_randomly(self, batch_size: int = None, queue_partial: bool = False, queue_index: int = -1) -> dict:
        items = []
        batch_size = batch_size or random.randint(5, 18)
        for i in range(batch_size):
            interval, data_size = self.sampler.sample()
            queue = self._generate_queue_by_rule(i, batch_size, queue_partial, queue_index)
            item = {
                "name": random.choice(self.WORKLOAD_TYPES),
                "interval": interval,
                "queue": queue,
                "dataSize": data_size
            }
            items.append(item)
        self.logger.info('Generated workloads with batch size %d.' % batch_size)

        return {'workloads': items}

    def _generate_queue_by_rule(self, current_index: int, total_num: int, queue_partial: bool, queue_index):
        if queue_index == 0:
            queue = choice_p(self.QUEUES, [3, 1])
        elif queue_index == 1:
            queue = random.choice(self.QUEUES)
        elif queue_index == 2:
            queue = choice_p(self.QUEUES, [1, 3])
        else:
            self.logger.error("Unexcept %d" % queue_index)
        return queue

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

    def load_evaluation_workloads(self,  first_n: int = -1, filename: str = None) -> dict:
        filename = filename or self.SAVE_FILENAME
        with open(filename, 'r') as f:
            workloads = json.load(f)
            if first_n > 0:
                workloads = {'workloads': workloads['workloads'][:first_n]}
            self.logger.info('Workloads %s loaded with %d items.' % (filename, len(workloads['workloads'])))
            return workloads

    def save_evaluation_workloads(self, workloads: dict, filename: str = None):
        with open(filename or self.SAVE_FILENAME, 'w') as f:
            json.dump(workloads, f)
            self.logger.info('Workloads %s saved.' % filename)


if __name__ == '__main__':
    gen = WorkloadGenerator()
    ws = [
        gen.generate_randomly(30, True, 0),
        gen.generate_randomly(30, True, 1),
        gen.generate_randomly(30, True, 2),
    ]

    for i, w in enumerate(ws):
        gen.save_evaluation_workloads(w, '../results/workload%d.json' % i)
