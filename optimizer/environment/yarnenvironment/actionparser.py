from ...hyperparameters import QUEUES


class ActionParser(object):

    def __init__(self):
        queue_names = QUEUES.get("names")
        actions = QUEUES.get("actions")
        self.action_space = len(actions)
        self.actions = {}
        for i in range(self.action_space):
            action = self._build_action(i, queue_names, actions)
            self.actions[i] = action

    def parse(self):
        return self.actions

    @staticmethod
    def _build_action(index: int, queue_names: list, actions: dict):
        action_list = actions.get(index)
        action = {}
        for name, s in zip(queue_names, action_list):
            action[name] = s
        return action
