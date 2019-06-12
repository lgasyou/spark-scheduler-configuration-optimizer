import json
from typing import Dict

import requests


def get_json(url: str) -> Dict[str, object]:
    r = requests.get(url)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return json.loads(r.text)
