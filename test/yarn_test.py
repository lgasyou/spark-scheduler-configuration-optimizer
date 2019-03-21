import unittest

import src.env.yarn as y
from src.env.sls_job_parser import GoogleTraceParser


class YarnTest(unittest.TestCase):

    def test_read_conf(self):
        com = y.YarnSchedulerCommunicator('/Users/xenon/Desktop/SLS')
        print([r.__dict__ for r in com.resources])
        print([r.__dict__ for r in com.constraints.queue])

    def test_json_reader(self):
        p = GoogleTraceParser('/Users/xenon/Desktop/cluster-scheduler-configuration-optimizer/googleTraceOutputDir')
        t = p.parse(1, 0, save_as_csv=False)
        print(t)
