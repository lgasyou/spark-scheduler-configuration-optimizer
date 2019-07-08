from optimizer.util import jsonutil
from optimizer.environment.yarnenvironment.jobfinishtimepredictor import Container

# finished_jobs = build_finished_jobs_from_json(job_json)
# time_costs = [j.elapsed_time for j in finished_jobs]
# print(time_costs, sum(time_costs))


# def containers(application_id: str) -> list:
#     job_json = jsonutil.get_json('http://localhost:8188/ws/v1/applicationhistory/apps/%s/appattempts' % application_id)
#     attempt_json = jsonutil.get_json(
#         'http://localhost:8188/ws/v1/applicationhistory/apps/%s/appattempts/%s/containers' %
#         (application_id, job_json['appAttempt'][0]['appAttemptId']))
#     print(attempt_json['container'][0])
#     return [Container(int(container_0['startedTime'])) for container_0 in attempt_json['container']]
#
#
# print([c.start_time for c in containers('application_1562578242447_0001')])

state = 'RUNNING' if 1 != 0 else 'FINISHED'
print(state)
