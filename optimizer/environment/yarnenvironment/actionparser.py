from optimizer.hyperparameters import QUEUES


class ActionParser(object):

    @staticmethod
    def parse():
        queue_names = QUEUES.get("names")
        raw_actions = QUEUES.get("actions")
        action_space = len(raw_actions)
        actions = {}
        for i in range(action_space):
            action = ActionParser._build_action(i, queue_names, raw_actions)
            actions[i] = action

        return actions

    @staticmethod
    def _build_action(index: int, queue_names: list, actions: dict):
        action_list = actions.get(index)
        action = {}
        for name, s in zip(queue_names, action_list):
            action[name] = s
        return action
