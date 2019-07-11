from optimizer.environment.spark.jobfinishtimepredictor import *

# container_0 = Container(0)
# container_0.add_tasks()

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

# containers = [container_0]
# print(predictor.simulate(8, [ContainerAddition(3, 1), ContainerAddition(6, 2), ContainerAddition(9, 1)], containers))
print(predictor.simulate([
    Container(0, 'RUNNING'),
    Container(3, 'RUNNING'),
    Container(6, 'RUNNING'),
    Container(6, 'RUNNING'),
    Container(9, 'RUNNING')
], [
    Task(1, 6),
    Task(2, 6),
    Task(3, 3),
    Task(4, 3),
    Task(5, 2),
    Task(6, 3),
    Task(7, 6),
]))
