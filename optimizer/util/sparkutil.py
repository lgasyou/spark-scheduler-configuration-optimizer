import os
import time

from optimizer.util import processutil


def spark_submit(workloads, spark_home, hadoop_home, java_home):
    workloads = workloads['workloads']
    for w in workloads:
        start_workload_process(w['name'], w['queue'], w['dataSize'],
                               os.getcwd(), spark_home, hadoop_home, java_home)
        time.sleep(w['interval'])


def start_workload_process(workload_type, queue, data_size, wd, spark_home, hadoop_home, java_home):
    c = ['%s/bin/start-spark-workload.sh' % wd, workload_type, spark_home, hadoop_home, java_home, wd, queue, data_size]
    return processutil.start_process(c)


def clean_spark_log(wd, hadoop_home):
    cmd = ['%s/bin/clean-spark-log.sh' % wd, hadoop_home]
    return processutil.start_process(cmd)
