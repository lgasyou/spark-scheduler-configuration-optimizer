import json

from optimizer.util import jsonutil

j = jsonutil.get_json('http://omnisky:8088/ws/v1/cluster/scheduler')
print(json.dumps(j))
