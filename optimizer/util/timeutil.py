import time


GMT_FORMAT = '%Y-%m-%dT%H:%M:%S.%fGMT'
TIME_ADDITION = 8 * 3600 * 1000


def convert_str_to_timestamp(time_str: str) -> int:
    ms = int(time_str.split('.')[1][:3])
    return int(time.mktime(time.strptime(time_str, GMT_FORMAT))) * 1000 + TIME_ADDITION + ms


def convert_timestamp_to_str(timestamp: int, fmt: str = None) -> str:
    time_array = time.localtime(timestamp / 1000)
    return time.strftime(fmt or '%Y-%m-%d %H:%M:%S', time_array)


def current_time_ms() -> int:
    return int(time.time() * 1000)
