import sys
import os
import inspect
from collections import defaultdict


class PathTracker:
    def __init__(self):
        self.paths = set()
        self.original_join = os.path.join

    def custom_join(self, *args):
        path = self.original_join(*args)
        self.paths.add(path)
        return path

    def print_report(self):
        print("\nPath Operations Report:")
        print("=====================")
        for path in sorted(self.paths):
            if os.path.exists(path):
                print(f"- {path}")


def run_with_tracking(script_path):
    tracker = PathTracker()
    os.path.join = tracker.custom_join

    with open(script_path) as f:
        code = compile(f.read(), script_path, 'exec')
        exec(code, {'__file__': script_path, '__name__': '__main__'})

    tracker.print_report()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python path_tracker.py <script.py>")
        sys.exit(1)

    run_with_tracking(sys.argv[1])