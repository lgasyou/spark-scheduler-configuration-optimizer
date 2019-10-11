import logging

from optimizer.environment.stateobtaining import yarnmodel


class RewardCalculator(object):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_delays = {}

    def get_reward(self, state: yarnmodel.State):
        waiting_apps = state.waiting_apps
        running_apps = state.running_apps
        apps = waiting_apps + running_apps

        if self.last_delays:
            percentages = []
            for app in apps:
                app_id = app.application_id
                predicted_delay = app.predicted_delay
                if app_id in self.last_delays and self.last_delays[app_id]:
                    last_predicted_delay = self.last_delays[app_id]
                    percentage = (last_predicted_delay - predicted_delay) / last_predicted_delay
                    percentages.append(percentage)

            average_percentage = sum(percentages) / len(percentages) if percentages else 0
        else:
            average_percentage = 0
        self.last_delays = {app.application_id: app.predicted_delay for app in apps}

        return average_percentage
