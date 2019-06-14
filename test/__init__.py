# import requests
#
#
# r = requests.get('http://localhost:18088/ws/v1/cluster/apps?states=RUNNING')
# print(r.text)

from optimizer.environment.yarnenvironment.actionparser import ActionParser

ap = ActionParser()
print(ap.actions)
print(ap.action_space)
