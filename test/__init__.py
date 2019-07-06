from optimizer.util import jsonutil
from optimizer.environment.yarnenvironment.sparkcommunicator import build_finished_jobs_from_json


job_json = jsonutil.get_json('http://localhost:18088/ws/v1/cluster/apps?states=FINISHED')
finished_jobs = build_finished_jobs_from_json(job_json)
time_costs = [j.elapsed_time for j in finished_jobs]
print(time_costs, sum(time_costs))
