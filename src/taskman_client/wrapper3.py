import os
import traceback

import requests



def report_run3(func):
    def func_wrapper(args):
        r = func(args)
        return r
    return func_wrapper

class JobContext:
    def __init__(self, run_name, suppress_inner_job=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass