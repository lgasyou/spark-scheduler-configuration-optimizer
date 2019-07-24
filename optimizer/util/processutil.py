import signal
import subprocess
import threading

import psutil


def has_process_finished(process: subprocess.Popen):
    return process is None or process.poll() is not None


def has_thread_finished(thread: threading.Thread):
    return thread is None or not thread.is_alive()


def kill_process_family(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        for process in children:
            process.send_signal(sig)
        parent.send_signal(sig)
    except psutil.NoSuchProcess:
        return


def kill_process_and_wait(process: subprocess.Popen):
    if process is not None:
        kill_process_family(process.pid)
        process.wait()


def start_process(cmd: list) -> subprocess.Popen:
    cmd = [str(c) for c in cmd]
    cmd_str = ' '.join(cmd)
    return subprocess.Popen(cmd_str, shell=True)
