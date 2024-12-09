import builtins
import sys
import os
import inspect
from collections import defaultdict
import sys
import os
import inspect
from collections import defaultdict

from cpath import project_root


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
        print("# Used files")

        for filename in sorted(self.call_counts.keys()):
            rel_path = os.path.relpath(filename, self.base_dir)
            print(rel_path)
            print("\nFunction calls:")
            for func, count in sorted(self.call_counts[filename].items()):
                print(f"  {func}: called {count} times")



def run_with_tracking(script_path):
    with open(script_path) as f:
        code = compile(f.read(), script_path, 'exec')
        exec(code, {'__file__': script_path, '__name__': '__main__'})


class FileTracker:
    def __init__(self):
        self.opened_files = set()
        self.original_open = open
        self.include_root = project_root

    def custom_open(self, file, *args, **kwargs):
        if isinstance(file, str):
            abs_path = os.path.abspath(file)
            if abs_path.startswith(self.include_root):
                self.opened_files.add(abs_path)
        return self.original_open(file, *args, **kwargs)

    def print_report(self):
        print("\nOpened Files Report:")
        print("===================")
        for filepath in sorted(self.opened_files):
            print(f"- {filepath}")
        print("===================")



def run_with_tracking(script_path, script_args):
    # Modify sys.argv to match the script's expected argv
    original_argv = sys.argv.copy()
    sys.argv = [script_path] + script_args

    f_tracker = FileTracker()
    builtins.__dict__['open'] = f_tracker.custom_open

    tracker = CodeTracker()
    sys.settrace(tracker.trace_calls)

    try:
        # Execute script with its arguments
        with open(script_path) as f:
            code = compile(f.read(), script_path, 'exec')
            global_vars = {
                '__file__': script_path,
                '__name__': '__main__'
            }
            exec(code, global_vars)
    finally:
        # Restore original argv and disable tracing
        sys.argv = original_argv
        sys.settrace(None)
        tracker.print_report()
        f_tracker.print_report()


def print_usage():
    print("Usage: python code_tracker.py <script.py> [script arguments...]")
    print("Example: python code_tracker.py my_script.py arg1 arg2")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()

    script_path = sys.argv[1]
    script_args = sys.argv[2:]  # All arguments after the script path

    if not os.path.exists(script_path):
        print(f"Error: Script '{script_path}' not found.")
        print_usage()

    run_with_tracking(script_path, script_args)