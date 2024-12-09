import sys
import os
import inspect
from collections import defaultdict


def run_with_tracking(script_path):
    tracker = FileTracker()
    builtins.__dict__['open'] = tracker.custom_open

    with open(script_path) as f:
        code = compile(f.read(), script_path, 'exec')
        exec(code, {'__file__': script_path, '__name__': '__main__'})

    tracker.print_report()


if __name__ == "__main__":
    import builtins

    if len(sys.argv) != 2:
        print("Usage: python file_tracker.py <script.py>")
        sys.exit(1)

    run_with_tracking(sys.argv[1])