import logging
import os
import time

from workloadsubmission import configrationmodel

algorithm_string = ["", "linear", "als", "kmeans", "svm", "bayes", "FPGrowth", "lda"]
algorithm_datafile = ["", "linear_gendata.txt", "matrix_kind2.txt", ""]
algorithm_workfile = ["", "gendata_"]


def start_workload_process(workload_type, queue, option, wd, spark_home, hadoop_home, java_home):
    c = '%s/bin/start-spark-workload.sh' % wd, workload_type, spark_home, hadoop_home, java_home, wd, queue, option
    return c


def workload_interface(load_dict, spark_home, hadoop_home, java_home):
    setting = configrationmodel.getitem(load_dict)
    for item in setting.item:
        command_string = "/home/lzq/library/spark-2.4.1-bin-hadoop2.7/bin/spark-submit --class " + setting.item[
            i].name + " --matster yarn  --deploy-mode cluster --queue " + setting.item[
                              i].queue + " --num-executors 5 --executor-cores 4  --executor-memory 6g --driver-memory 1g /home/lzq/spark_jar/workload_" + \
                          setting.item[i].name + ".jar " + str(setting.item[i].data_size)
        logging.info('{}: {}'.format(i, command_string))

        start_workload_process(item.name, item.queue, item.data_size, os.getcwd(), spark_home, hadoop_home, java_home)
        # subprocess.Popen(command_string, shell=True)
        time.sleep(item.interval)
