from optimizer.environment.spark.sparkcommunicator import SparkWorkloadController, SparkCommunicator, restart_yarn

import os
import time

this = []
for _ in range(1):
    controller = SparkWorkloadController()
    controller.start_workloads(os.getcwd() + '/../', '/home/lzq/library/spark-2.4.1-bin-hadoop2.7',
                               '/home/lzq/library/hadoop', '/home/lzq/library/jdk1.8.0_211')
    while not controller.is_done():
        print('waiting')
        time.sleep(1)

    _1, _2 = SparkCommunicator.get()
    this.append((_1, _2))
    print(_1, _2)

    restart_yarn(os.getcwd() + '/../', '/home/lzq/library/hadoop').wait()


print('done')
for _1, _2 in this:
    print(_1, _2)
