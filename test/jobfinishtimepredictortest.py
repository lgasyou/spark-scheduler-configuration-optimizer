from optimizer.environment.yarnenvironment.jobfinishtimepredictor import *

container_0 = Container(0)
container_0.add_tasks([
    Task(1, 6, 6),
    Task(2, 6, 12),
    Task(3, 3, 15),
    Task(4, 3, 18),
    Task(5, 2, 20),
    Task(6, 3, 23),
    Task(7, 6, 29),
])

predictor = JobFinishTimePredictor()
# print(predictor.predict(2, containers))
# print([[task.id for task in c.tasks] for c in containers])
#
# containers.extend([Container(7), Container(7)])
# print(predictor.predict(6, containers))
# print([[task.id for task in c.tasks] for c in containers])
#
# containers.append(Container(9))
# print(predictor.predict(8, containers))

containers = [container_0]
print(predictor.simulate(8, [ContainerAddition(3, 1), ContainerAddition(6, 2), ContainerAddition(9, 1)], containers))
print([[task.id for task in c.tasks] for c in containers])
