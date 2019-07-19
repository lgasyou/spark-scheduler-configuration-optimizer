import os

from optimizer.util import processutil


def restart_yarn(wd, hadoop_home):
    cmd = ["%s/bin/restart-yarn.sh" % wd, hadoop_home]
    return processutil.start_process(cmd)


def refresh_queues(hadoop_home: str):
    cmd = [os.path.join(os.getcwd(), 'bin', 'refresh-queues.sh'), hadoop_home]
    return processutil.start_process(cmd)
