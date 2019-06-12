import requests


r = requests.get('http://localhost:18088/ws/v1/cluster/apps?states=RUNNING')
print(r.text)
