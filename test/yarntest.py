import logging

import src.env.yarn as y


def communicator_test():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    sls_jobs_json = '/data/testset/' + 'sls-jobs.json'
    com = y.YarnCommunicator(
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
