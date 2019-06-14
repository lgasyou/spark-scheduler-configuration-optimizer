# import requests
#
#
# r = requests.get('http://localhost:18088/ws/v1/cluster/apps?states=RUNNING')
# print(r.text)

from optimizer.environment.yarnenvironment.actionparser import ActionParser
from optimizer.environment.yarnenvironment.schedulerstrategy import SchedulerStrategyFactory

ap = ActionParser()
actions = ap.parse()

f = SchedulerStrategyFactory()
s = f.create("CapacityScheduler", "", "", actions)
print(s.action_set)
