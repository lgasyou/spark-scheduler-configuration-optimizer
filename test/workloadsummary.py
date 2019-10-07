import requests
import json
from typing import Dict


def get_json(url: str) -> Dict[str, object]:
    r = requests.get(url)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return json.loads(r.text)


rm_api = 'http://10.1.114.60:8088/'


apps = get_json(rm_api + 'ws/v1/cluster/apps')['apps']['app']
apps.sort(key=lambda a: a['id'])
for app in apps:
    print(app['name'], app['elapsedTime'])
