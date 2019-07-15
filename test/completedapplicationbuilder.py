import dataclasses

from optimizer.util import jsonutil, timeutil


@dataclasses.dataclass
class CompletedApplication(object):
    start_time: int
    end_time: int
    elapsed_time: int
    name: int

    def __str__(self):
        return '{:<8} {} {}'.format(
            self.name,
            timeutil.convert_timestamp_to_str(self.start_time, '%H:%M:%S'),
            timeutil.convert_timestamp_to_str(self.end_time, '%H:%M:%S')
        )


class CompletedApplicationBuilder(object):

    def __init__(self):
        self.url = 'http://omnisky:8188/'

    def build(self, application_id: str):
        json: dict = jsonutil.get_json(self.url + 'ws/v1/applicationhistory/apps/%s/' % application_id)
        start_time = json['startedTime']
        end_time = json['finishedTime']
        elapsed_time = json['elapsedTime']
        name = json['name']
        return CompletedApplication(start_time, end_time, elapsed_time, name)
