import requests
import json


r = requests.get('http://localhost:18088/ws/v1/cluster/apps?states=RUNNING')
d = eval(r.text.replace('false', 'False').replace('true', 'True'))
with open('../results/dump.json', 'w') as f:
    json.dump(d, f)
    print(d)
