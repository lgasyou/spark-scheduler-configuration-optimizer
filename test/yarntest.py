import json
import unittest

import requests

import src.env.yarn as y
from src.env.slsjobparser import GoogleTraceParser


class YarnTest(unittest.TestCase):

    def test_read_conf(self):
        com = y.YarnSchedulerCommunicator(
            'http://localhost:8088',
            '/Users/xenon/Desktop/SLS'
        )
        state = com.get_state()
        resources = state.resources
        constraint = state.constraint
        queue_c = constraint.queue
        print([r.__dict__ for r in resources])
        print([r.__dict__ for r in queue_c])

    def test_json_reader(self):
        p = GoogleTraceParser('/Users/xenon/Desktop/cluster-scheduler-configuration-optimizer/googleTraceOutputDir')
        t = p.parse(1, 0, save_as_csv=False)
        print(t)

    def test_connection(self):
        r = requests.get('http://baidu.com', headers={
            'Content-Type': 'application/json'
        })
        print(r.headers)

    def test_write(self):
        s = '{"scheduler":{"schedulerInfo":{"type":"capacityScheduler","capacity":100.0,"usedCapacity":0.0,"maxCapacity":100.0,"queueName":"root","queues":{"queue":[{"type":"capacitySchedulerLeafQueueInfo","capacity":50.0,"usedCapacity":0.0,"maxCapacity":50.0,"absoluteCapacity":50.0,"absoluteMaxCapacity":50.0,"absoluteUsedCapacity":0.0,"numApplications":0,"queueName":"queueA","state":"RUNNING","resourcesUsed":{"memory":0,"vCores":0},"hideReservationQueues":false,"nodeLabels":["*"],"allocatedContainers":0,"reservedContainers":0,"pendingContainers":0,"capacities":{"queueCapacitiesByPartition":[{"partitionName":"","capacity":50.0,"usedCapacity":0.0,"maxCapacity":50.0,"absoluteCapacity":50.0,"absoluteUsedCapacity":0.0,"absoluteMaxCapacity":50.0,"maxAMLimitPercentage":20.0}]},"resources":{"resourceUsagesByPartition":[{"partitionName":"","used":{"memory":0,"vCores":0},"reserved":{"memory":0,"vCores":0},"pending":{"memory":0,"vCores":0},"amUsed":{"memory":0,"vCores":0},"amLimit":{"memory":0,"vCores":0},"userAmLimit":{"memory":0,"vCores":0}}]},"numActiveApplications":0,"numPendingApplications":0,"numContainers":0,"maxApplications":5000,"maxApplicationsPerUser":5000,"userLimit":100,"users":null,"userLimitFactor":1.0,"AMResourceLimit":{"memory":0,"vCores":0},"usedAMResource":{"memory":0,"vCores":0},"userAMResourceLimit":{"memory":0,"vCores":0},"preemptionDisabled":false,"intraQueuePreemptionDisabled":true,"defaultPriority":0},{"type":"capacitySchedulerLeafQueueInfo","capacity":50.0,"usedCapacity":0.0,"maxCapacity":50.0,"absoluteCapacity":50.0,"absoluteMaxCapacity":50.0,"absoluteUsedCapacity":0.0,"numApplications":0,"queueName":"queueB","state":"RUNNING","resourcesUsed":{"memory":0,"vCores":0},"hideReservationQueues":false,"nodeLabels":["*"],"allocatedContainers":0,"reservedContainers":0,"pendingContainers":0,"capacities":{"queueCapacitiesByPartition":[{"partitionName":"","capacity":50.0,"usedCapacity":0.0,"maxCapacity":50.0,"absoluteCapacity":50.0,"absoluteUsedCapacity":0.0,"absoluteMaxCapacity":50.0,"maxAMLimitPercentage":20.0}]},"resources":{"resourceUsagesByPartition":[{"partitionName":"","used":{"memory":0,"vCores":0},"reserved":{"memory":0,"vCores":0},"pending":{"memory":0,"vCores":0},"amUsed":{"memory":0,"vCores":0},"amLimit":{"memory":0,"vCores":0},"userAmLimit":{"memory":0,"vCores":0}}]},"numActiveApplications":0,"numPendingApplications":0,"numContainers":0,"maxApplications":5000,"maxApplicationsPerUser":5000,"userLimit":100,"users":null,"userLimitFactor":1.0,"AMResourceLimit":{"memory":0,"vCores":0},"usedAMResource":{"memory":0,"vCores":0},"userAMResourceLimit":{"memory":0,"vCores":0},"preemptionDisabled":false,"intraQueuePreemptionDisabled":true,"defaultPriority":0}]},"capacities":{"queueCapacitiesByPartition":[{"partitionName":"","capacity":100.0,"usedCapacity":0.0,"maxCapacity":100.0,"absoluteCapacity":100.0,"absoluteUsedCapacity":0.0,"absoluteMaxCapacity":100.0,"maxAMLimitPercentage":0.0}]},"health":{"lastrun":0,"operationsInfo":{"entry":{"key":"last-preemption","value":{"nodeId":"N/A","containerId":"N/A","queue":"N/A"}},"entry":{"key":"last-reservation","value":{"nodeId":"N/A","containerId":"N/A","queue":"N/A"}},"entry":{"key":"last-allocation","value":{"nodeId":"N/A","containerId":"N/A","queue":"N/A"}},"entry":{"key":"last-release","value":{"nodeId":"N/A","containerId":"N/A","queue":"N/A"}}},"lastRunDetails":[{"operation":"releases","count":0,"resources":{"memory":0,"vCores":0}},{"operation":"allocations","count":0,"resources":{"memory":0,"vCores":0}},{"operation":"reservations","count":0,"resources":{"memory":0,"vCores":0}}]}}}}'
        j = json.loads(s)
        with open('cluster-scheduler.json', 'w') as f:
            json.dump(j, f)
