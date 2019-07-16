from optimizer.environment.spark import sparkmodel
from optimizer.util import jsonutil, timeutil


class SparkApplicationBuilder(object):

    def __init__(self, spark_history_server_api_url):
        self.spark_history_server_api_url = spark_history_server_api_url
        self.application_id = None

    def build(self, application_id: str):
        self.application_id = application_id
        return self.parse_and_build_application()

    # noinspection PyTypeChecker
    def parse_and_build_application(self):
        application_url = self.spark_history_server_api_url + 'applications/%s/1/' % self.application_id
        application_json = jsonutil.get_json(application_url)
        start_time = timeutil.convert_str_to_timestamp(application_json['startTime'])

        jobs_url = self.spark_history_server_api_url + 'applications/%s/1/jobs' % self.application_id
        jobs_json = jsonutil.get_json(jobs_url)
        jobs = [self.parse_and_build_job(job_json) for job_json in jobs_json]
        jobs.reverse()

        try:
            input_bytes = jobs[0].stages[0].input_bytes
        except IndexError:
            input_bytes = 0

        executors_url = self.spark_history_server_api_url + 'applications/%s/1/allexecutors' % self.application_id
        executors_json = jsonutil.get_json(executors_url)
        executors = [
            self.parse_and_build_executor(executor_json)
            for executor_json in executors_json if executor_json['id'] != 'driver'
        ]

        return sparkmodel.Application(self.application_id, start_time, jobs, executors, input_bytes)

    def parse_and_build_job(self, j):
        job_id = j['jobId']
        name = j['name']
        stage_ids = j['stageIds']
        stages = []
        for stage_id in stage_ids:
            stage_url = self.spark_history_server_api_url + 'applications/%s/1/stages/%s' % (
                self.application_id, stage_id)
            stage_json = jsonutil.get_json(stage_url)
            stage = self.parse_and_build_stage(stage_json)
            stages.append(stage)
        stages.sort(key=lambda s: s.stage_id)
        return sparkmodel.Job(job_id, name, stages)

    def parse_and_build_stage(self, j):
        j = j[0]
        stage_id = j['stageId']
        num_tasks = j['numTasks']
        input_bytes = j['inputBytes']
        name = j['name']
        tasks_url = self.spark_history_server_api_url + 'applications/%s/1/stages/%s/0/taskList' % (
            self.application_id, stage_id)
        tasks_json: dict = jsonutil.get_json(tasks_url)
        tasks = [self.parse_and_build_task(task_json) for task_json in tasks_json]

        return sparkmodel.Stage(stage_id, num_tasks, input_bytes, name, tasks)

    @staticmethod
    def parse_and_build_task(j):
        task_id = j['taskId']
        launch_time = timeutil.convert_str_to_timestamp(j['launchTime'])
        duration = j.get('duration', 0)
        host = j['host']
        if 'taskMetrics' in j and 'inputMetrics' in j['taskMetrics'] and 'bytesRead' in j['taskMetrics']['inputMetrics']:
            input_bytes = j['taskMetrics']['inputMetrics']['bytesRead']
        else:
            input_bytes = 0
        return sparkmodel.Task(task_id, launch_time, duration, host, input_bytes)

    @staticmethod
    def parse_and_build_executor(j):
        executor_id = int(j['id'])
        is_active = j['isActive']
        start_time = timeutil.convert_str_to_timestamp(j['addTime'])
        max_memory_gb = j['maxMemory'] / 1024**3
        return sparkmodel.Executor(executor_id, is_active, start_time, max_memory_gb)
