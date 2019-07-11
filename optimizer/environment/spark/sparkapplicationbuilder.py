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

        executors_url = self.spark_history_server_api_url + 'applications/%s/1/allexecutors' % self.application_id
        executors_json = jsonutil.get_json(executors_url)
        executors = [
            self.parse_and_build_executor(executor_json)
            for executor_json in executors_json if executor_json['id'] != 'driver'
        ]

        return sparkmodel.Application(self.application_id, start_time, jobs, executors)

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
        return sparkmodel.Job(job_id, name, stages)

    def parse_and_build_stage(self, j):
        stage_id = j[0]['stageId']
        num_tasks = j[0]['numTasks']
        name = j[0]['name']
        tasks_url = self.spark_history_server_api_url + 'applications/%s/1/stages/%s/0/taskList' % (
            self.application_id, stage_id)
        tasks_json = jsonutil.get_json(tasks_url)
        tasks = [self.parse_and_build_task(task_json) for task_json in tasks_json]
        return sparkmodel.Stage(stage_id, num_tasks, name, tasks)

    @staticmethod
    def parse_and_build_task(j):
        task_id = j['taskId']
        launch_time = j['launchTime']
        duration = j['duration']
        host = j['host']
        return sparkmodel.Task(task_id, launch_time, duration, host)

    @staticmethod
    def parse_and_build_executor(j):
        executor_id = int(j['id'])
        is_active = j['isActive']
        start_time = timeutil.convert_str_to_timestamp(j['addTime'])
        max_memory_gb = j['maxMemory'] / 1024**3
        return sparkmodel.Executor(executor_id, is_active, start_time, max_memory_gb)
