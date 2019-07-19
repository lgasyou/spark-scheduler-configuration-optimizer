import os

from optimizer.util import processutil, jsonutil


def restart_yarn(wd, hadoop_home):
    cmd = ["%s/bin/restart-yarn.sh" % wd, hadoop_home]
    return processutil.start_process(cmd)


def refresh_queues(hadoop_home: str):
    cmd = [os.path.join(os.getcwd(), 'bin', 'refresh-queues.sh'), hadoop_home]
    return processutil.start_process(cmd)


def has_all_application_done(rm_api: str) -> bool:
    url = rm_api + 'ws/v1/cluster/apps?states=NEW,NEW_SAVING,SUBMITTED,ACCEPTED,RUNNING'
    unfinished_app_json = jsonutil.get_json(url)
    all_app_finished = unfinished_app_json['apps'] is None

    url = rm_api + 'ws/v1/cluster/apps?states=FINISHED'
    finished_app_json = jsonutil.get_json(url)
    has_finished_app = finished_app_json['apps'] is not None

    return all_app_finished and has_finished_app
