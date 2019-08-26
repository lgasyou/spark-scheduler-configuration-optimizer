import logging
import threading
import time

from optimizer.environment.stateobtaining.statebuilder import StateBuilder
from optimizer.util import timeutil


class RegularDelayFetcher(object):

    def __init__(self, state_builder: StateBuilder):
        self.logger = logging.getLogger(__name__)
        self.state_builder = state_builder
        self.running = False
        self.last_run_time = 0
        self.delays = {}

    def get_interval_average_delay(self):
        apps = self.state_builder.parse_and_build_finished_apps()
        delays = [a.elapsed_time for a in apps if a.finished_time > self.last_run_time]
        return sum(delays) / len(delays) if len(delays) else 0

    def start_heartbeat(self, every_n_seconds: int = 180):
        if self.running:
            self.logger.warning('There is already a thread running. Will not start another one.')
            return

        self.delays, self.running = {}, True
        thread = threading.Thread(target=self._regularly_get_delay, args=(every_n_seconds,))
        thread.start()
        self.logger.info('RegularDelayFetcher started.')

    def stop(self):
        self.running = False

    def save_delays(self, filename: str):
        with open(filename, 'w') as f:
            f.write(str(self.delays))
            self.logger.info('Delay file %s saved.' % filename)

    def _regularly_get_delay(self, seconds: int):
        timer, t = 0, 0
        while self.running:
            if timer % seconds == 0:
                # noinspection PyBroadException
                try:
                    time_delay = self.get_interval_average_delay()
                    self.delays[t] = time_delay
                    self.logger.info('Time %d: Average Delay: %f.' % (t, time_delay))
                    t += 1
                    self.last_run_time = timeutil.current_time_ms()
                except Exception:
                    self.logger.exception("Exception when regularly gets delay. Sleep %d seconds" % seconds)
            timer += 1
            time.sleep(1)

        self.logger.info('Delay thread exited.')
