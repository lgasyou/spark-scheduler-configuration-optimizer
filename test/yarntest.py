import logging
import unittest

import src.env.yarn as y


class YarnTest(unittest.TestCase):

    def test_read_conf(self):
        com = y.YarnSchedulerCommunicator(
            'http://localhost:8088',
            '/Users/xenon/Desktop/SLS',
            ''
        )
        state = com.get_state()
        resources = state.resources
        constraint = state.constraint
        queue_c = constraint.queue
        print([r.__dict__ for r in resources])
        print([r.__dict__ for r in queue_c])

    def test_yield(self):
        def y1():
            for i in range(10):
                yield i

        def y2():
            for i in range(10):
                yield y1()

        g = y2()
        for i in g:
            print(i)

    def test_state(self):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        sls_jobs_json = '../data/testset/' + 'sls-jobs.json'
        com = y.YarnSchedulerCommunicator(
            'http://localhost:18088/',
            '/opt/hadoop',
            sls_jobs_json
        )
        com.reset('../')
        while True:
            com.act(1)


# if __name__ == '__main__':
#     def y1():
#         for i in range(10):
#             yield i
#
#
#     def y2():
#         for i in range(10):
#             yield y1()
#
#
#     g = y2()
#     for i in g:
#         for j in i:
#             print(j)
#
#     import time
#     # 1553738104.193
#     # 1553738161.011
#     # 1553738228.526
#     print(time.time())

def communicator_test():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    sls_jobs_json = '/data/testset/' + 'sls-jobs.json'
    com = y.YarnSchedulerCommunicator(
        'http://localhost:18088/',
        '/opt/hadoop',
        sls_jobs_json
    )
    com.reset('../')
    while True:
        com.get_state_tensor()
        com.act(1)


if __name__ == '__main__':
    communicator_test()
