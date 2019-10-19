from optimizer.environment.stateobtaining import yarnmodel


class SimulationStateBuilder(object):

    @staticmethod
    def build(state_dict: dict) -> yarnmodel.State:
        waiting_apps = []
        for wa in state_dict['waitJob']:
            waiting_apps.append(yarnmodel.YarnApplication(wa['queue'], wa['container'], wa['id'], wa['worktime']))

        running_apps = []
        for ra in state_dict['runJob']:
            running_apps.append(yarnmodel.YarnApplication(ra['queue'], ra['container'], ra['id'], ra['worktime']))

        constraints = []
        for constraint in state_dict['stricts']:
            constraints.append(yarnmodel.QueueConstraint(constraint['common'], constraint['max']))

        return yarnmodel.State(waiting_apps, running_apps, constraints)
