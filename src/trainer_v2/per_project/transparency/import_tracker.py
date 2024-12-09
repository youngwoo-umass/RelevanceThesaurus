import sys
import os
import inspect
from collections import defaultdict


class CodeTracker:
    def __init__(self):
        self.call_counts = defaultdict(lambda: defaultdict(int))
        self.line_counts = defaultdict(lambda: defaultdict(int))
        self.base_dir = os.getcwd()

    def trace_calls(self, frame, event, arg):
        if event != 'call':
            return None

        code = frame.f_code
        filename = code.co_filename

        # Only track files in current project directory
        if not filename.startswith(self.base_dir):
            return None

        function_name = code.co_name
        self.call_counts[filename][function_name] += 1

        return self.trace_lines

    def trace_lines(self, frame, event, arg):
        if event != 'line':
            return

        filename = frame.f_code.co_filename
        if not filename.startswith(self.base_dir):
            return

        lineno = frame.f_lineno
        self.line_counts[filename][lineno] += 1

    def print_report(self):
        print("\nCode Execution Report")
        print("===================")

        for filename in sorted(self.call_counts.keys()):
            rel_path = os.path.relpath(filename, self.base_dir)
            print(f"\nFile: {rel_path}")

            print("\nFunction calls:")
            for func, count in sorted(self.call_counts[filename].items()):
                print(f"  {func}: called {count} times")
            #
            # print("\nLine-by-line execution:")
            # if filename in self.line_counts:
            #     with open(filename, 'r') as f:
            #         lines = f.readlines()
            #
            #     for i, line in enumerate(lines, 1):
            #         count = self.line_counts[filename].get(i, 0)
            #         print(f"  {count:4d} | {line.rstrip()}")


def run_with_tracking(script_path):
    tracker = CodeTracker()
    sys.settrace(tracker.trace_calls)

    # Execute script
    with open(script_path) as f:
        code = compile(f.read(), script_path, 'exec')
        global_vars = {
            '__file__': script_path,
            '__name__': '__main__'
        }
        exec(code, global_vars)

    sys.settrace(None)
    tracker.print_report()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python code_tracker.py <script.py>")
        sys.exit(1)

    run_with_tracking(sys.argv[1])