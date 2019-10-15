import time
import requests
import json
import subprocess
from typing import Dict


def get_json(url: str) -> Dict[str, object]:
    r = requests.get(url)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return json.loads(r.text)


def has_all_application_done(rm_api: str) -> bool:
    url = rm_api + 'ws/v1/cluster/apps?states=NEW,NEW_SAVING,SUBMITTED,ACCEPTED,RUNNING'
    unfinished_app_json = get_json(url)
    all_app_finished = unfinished_app_json['apps'] is None

    url = rm_api + 'ws/v1/cluster/apps?states=FINISHED'
    finished_app_json = get_json(url)
    has_finished_app = finished_app_json['apps'] is not None

    return all_app_finished and has_finished_app


def restart_yarn(wd, hadoop_home):
    cmd = ["%s/bin/restart-yarn.sh" % wd, hadoop_home]
    return start_process(cmd)


def start_process(cmd: list) -> subprocess.Popen:
    cmd = [str(c) for c in cmd]
    cmd_str = ' '.join(cmd)
    return subprocess.Popen(cmd_str, shell=True)


def start_workload_process(workload_type, queue, data_size, spark_home, hadoop_home, java_home):
    c = ['../bin/start-spark-workload.sh', workload_type,
         spark_home, hadoop_home, java_home,
         queue, data_size, '/home/ls/spark-scheduler-configuration-optimizer']
    return start_process(c)


def run_all_workloads():
    restart_yarn('..', hadoop_home).wait()
    print('Total %d' % len(workload_types) * len(data_sizes))
    for workload_type in workload_types:
        for data_size in data_sizes:
            start_workload_process(workload_type, 'queueB', data_size, spark_home, hadoop_home, java_home)
            time.sleep(90)
            while not has_all_application_done(rm_api):
                time.sleep(1)
            print('{} which uses {} data size finished.'.format(workload_type, data_size))


def summary():
    apps = get_json(rm_api + 'ws/v1/cluster/apps')['apps']['app']
    apps.sort(key=lambda a: a['id'])
    app_name = apps[0]
    data_size = 0
    for app in apps:
        if app_name == app['name']:
            data_size += 1
        else:
            app_name = app['name']
            data_size = 1

        print(app['name'].split('_')[-1], app['elapsedTime'], data_size, sep=',')


data_sizes = [str(i) for i in range(1, 9)]
# workload_types = ['bayes', 'FPGrowth', 'kmeans', 'lda', 'linear', 'svm',
#                   'rnn', 'autoencoder', 'lenet', ]
workload_types = ['resnet', 'vgg']
java_home = '/home/ls/library/jdk'
spark_home = '/home/ls/library/spark'
hadoop_home = '/home/ls/library/hadoop'
rm_api = 'http://10.1.114.60:8088/'

run_all_workloads()
summary()
