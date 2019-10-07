import logging
import os
import threading
import time

from optimizer.util import processutil


def async_start_workloads(workloads, spark_home, hadoop_home, java_home, wait: int = 30):
    args = (workloads, spark_home, hadoop_home, java_home)
    start_thread = threading.Thread(target=start_workloads, args=args)
    start_thread.start()
    logging.info("Wait for workloads start for %d seconds." % wait)
    time.sleep(wait)
    return start_thread


def start_workloads(workloads, spark_home, hadoop_home, java_home):
    workloads = workloads['workloads']
    for w in workloads:
        logging.info(w)
        start_workload_process(w['name'], w['queue'], w['dataSize'],
                               spark_home, hadoop_home, java_home)
        time.sleep(w['interval'])


def start_workload_process(workload_type, queue, data_size, spark_home, hadoop_home, java_home):
    c = ['bin/start-spark-workload.sh', workload_type,
         spark_home, hadoop_home, java_home,
         queue, data_size, os.getcwd()]
    return processutil.start_process(c)
