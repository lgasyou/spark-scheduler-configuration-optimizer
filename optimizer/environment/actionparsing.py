from optimizer.hyperparameters import QUEUES


def parse(scheduler_type: str):
    queue_names = QUEUES.get("names")
    raw_actions = QUEUES.get("actions")[scheduler_type]
    build_func = ACTION_BUILD_MAP[scheduler_type]
    return _build_actions(queue_names, raw_actions, build_func)


def _build_actions(queue_names, raw_actions, build_func):
    action_space = len(raw_actions)
    actions = {}
    for i in range(action_space):
        action = build_func(i, queue_names, raw_actions)
        actions[i] = action

    return actions


def _build_capacity_scheduler_action(index: int, queue_names: list, actions: dict):
    capacities, max_capacities = actions.get(index)
    action = {}
    for name, c, mc in zip(queue_names, capacities, max_capacities):
        action[name] = (c, mc)
    return action


def _build_fair_scheduler_action(index: int, queue_names: list, actions: dict):
    weights, scheduling_policy = actions.get(index)
    action = {}
    for name, weight in zip(queue_names, weights):
        action[name] = weight
    return {
        'weights': action,
        'schedulingPolicy': scheduling_policy
    }


ACTION_BUILD_MAP = {
    'CapacityScheduler': _build_capacity_scheduler_action,
    'FairScheduler': _build_fair_scheduler_action
}
