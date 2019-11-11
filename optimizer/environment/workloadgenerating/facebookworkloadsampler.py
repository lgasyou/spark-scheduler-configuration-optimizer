import logging
import os
import random


class FacebookWorkloadSampler(object):

    SAMPLE_FILENAMES = [
        'samples_with_queues.tsv',
        'FB-2009_samples_24_times_1hr_0.tsv',
        'FB-2009_samples_24_times_1hr_1.tsv',
        'FB-2010_samples_24_times_1hr_0.tsv',
        'FB-2010_samples_24_times_1hr_withInputPaths_0.tsv'
    ]

    def __init__(self, sampler_index: int, directory: str = 'data/facebook-samples'):
        logging.info('Using sample file: %s' % self.SAMPLE_FILENAMES[sampler_index])
        filename = os.path.join(directory, self.SAMPLE_FILENAMES[sampler_index])
        with open(filename, 'r') as f:
            self.samples = f.readlines()
            self.num_samples = len(self.samples)

    def sample(self):
        while True:
            line_num = random.randint(0, self.num_samples - 1)
            sample = self.samples[line_num]
            raw_interval, data_size, queue = self._parse_line(sample)
            interval = raw_interval if raw_interval < 100 else 100
            return interval, str(data_size), int(queue)

    def get_all_samples(self):
        return [self._parse_line(line) for line in self.samples]

    @staticmethod
    def _parse_line(line: str):
        data = line.split('\t')
        interval = int(data[2])
        data_size = (len(data[3]) + 1) // 2
        queue = data[6]
        return interval, data_size, queue

# data size statistic
# [0, 86, 752, 717, 1072, 1730, 208, 222, 343, 239, 249, 234, 41, 1, 0, 0, 0, 0, 0, 0]
# [0, 2, 1, 6, 2, 10, 18, 6, 2, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# [0, 183, 799, 829, 864, 1934, 251, 245, 332, 432, 343, 362, 62, 2, 0, 0, 0, 0, 0, 0]
# [0, 1126, 35, 2360, 1687, 2427, 2250, 2304, 3359, 2222, 3570, 1677, 1145, 278, 2, 0, 0, 0, 0, 0]
# [0, 1746, 20, 2729, 1792, 2242, 2471, 2244, 2794, 2165, 3931, 1713, 1337, 239, 5, 0, 0, 0, 0, 0]

# interval statistic
# [0, 3550, 2285, 58, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# [0, 9, 31, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# [0, 4108, 2470, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# [0, 22423, 2016, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# [0, 23365, 2062, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
