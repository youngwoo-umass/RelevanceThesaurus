import os
import time
import uuid
from typing import Dict

from taskman_client.RESTProxy import RESTProxy
from taskman_client.host_defs import webtool_host, webtool_port


class TaskManagerProxy(RESTProxy):
    def __init__(self, host, port):
        super(TaskManagerProxy, self).__init__(host, port)

    def task_update(self, run_name, uuid, tpu_name, machine, update_type, msg, job_id=None):
       pass

    def report_number(self, name, value, condition, field):
        pass


    def get_tpu(self, tpu_condition=None, uuid=None, run_name=""):
        pass

    def get_num_active_jobs(self, machine):
        data = {
            'machine_name': machine,
        }
        r = self.post("/task/get_num_active_jobs", data)
        return r['num_active_jobs']

    def get_num_pending_jobs(self, machine):
        data = {
            'machine_name': machine,
        }
        r = self.post("/task/get_num_pending_jobs", data)
        return r['num_pending_jobs']

    def pool_job(self, job_name, max_job, machine) -> int:
        data = {
            'max_job': max_job,
            'job_name': job_name,
            'machine': machine,
        }
        r = self.post("/task/pool_job", data, timeout=30)
        return r['job_id']

    def query_job_group_status(self, job_name) -> Dict:
        data = {
            'job_name': job_name,
        }
        r = self.post("/task/query_job_group_status", data)
        return r

    def query_task_status(self, run_name) -> Dict:
        data = {
            'run_name': run_name,
        }
        r = self.post("/task/query_task_status", data)
        return r

    def report_done_and_pool_job(self, job_name, max_job, machine, job_id) -> int:
        data = {
            'max_job': max_job,
            'job_name': job_name,
            'machine': machine,
            'job_id': job_id
        }
        r = self.post("/task/sub_job_done_and_pool_job", data, timeout=30)
        return r['job_id']

    def sub_job_update(self, job_name, machine, update_type, msg, job_id, max_job=None):
        data = {
            'job_name': job_name,
            'machine': machine,
            'update_type': update_type,
            'msg': msg
        }
        if max_job is not None:
            data['max_job'] = max_job
        if job_id is not None:
            data['job_id'] = job_id
        return self.post("/task/sub_job_update", data)

    def cancel_allocation(self, job_name, job_id):
        data = {
            'job_name': job_name,
            'job_id': job_id
        }
        r = self.post("/task/cancel_allocation", data)
        return r['job_id']

    def make_success_notification(self):
        data = {
            'msg': "SUCCESSFUL_TERMINATE",
        }
        print(data)
        r = self.post("/task/make_notification", data)
        return r

    def make_fail_notification(self):
        data = {
            'msg': "ABNORMAL_TERMINATE",
        }
        print(data)
        r = self.post("/task/make_notification", data)
        return r



def get_local_machine_name():
    if os.name == "nt":
        return os.environ["COMPUTERNAME"]
    else:
        return os.uname()[1]



def get_task_manager_proxy():
    return TaskManagerProxy(webtool_host, webtool_port)


def assign_tpu_anonymous(wait=True):
    print("Auto assign TPU")
    condition = get_applicable_tpu_condition()

    def request_tpu():
        return get_task_manager_proxy().get_tpu(condition)['tpu_name']

    assigned_tpu = request_tpu()

    sleep_time = 5
    while wait and assigned_tpu is None:
        time.sleep(sleep_time)
        if sleep_time < 300:
            sleep_time += 10
        assigned_tpu = request_tpu()
    print("Assigned tpu : ", assigned_tpu)
    return assigned_tpu


def get_applicable_tpu_condition():
    machine = get_local_machine_name()
    if machine in ["lesterny", "instance-5", "us-1"]:
        condition = "v2"
    elif machine == "instance-3":
        condition = "v3"
    elif machine == "instance-4":
        condition = "v3"
    else:
        print("Warning TPU condition not indicated")
        condition = None
    return condition

