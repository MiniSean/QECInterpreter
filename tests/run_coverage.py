# -------------------------------------------
# Coverage collection of test suit
# Expects all test scripts to match:
# 'test_*.py'
# -------------------------------------------
import os
import coverage
import webbrowser
import subprocess
from typing import Union, Callable, Any
from unittest import TestLoader, TestResult, TextTestRunner
from pathlib import Path

# Use resolve() to get an absolute path
TEST_DIR = str(Path(__file__).resolve().parent)
REP_DIR = os.path.join(TEST_DIR, 'coverage_report')
IND_DIR = os.path.join(REP_DIR, 'index.html')


# region Module Methods
def write_coverage_report() -> Callable[[], Any]:
    """
    Instantiates coverage object, calls code, returns coverage report.
    Returns callable that opens coverage report in browser.
    """
    cov = coverage.Coverage()
    cov.start()

    # Call test code
    run_tests(test_directory=TEST_DIR)

    cov.stop()
    cov.save()

    prepare_path(global_path=REP_DIR, make_dir=True)
    cov.html_report(directory=REP_DIR)
    return load_coverage_report


def run_tests(test_directory: str):
    """Runs all test scripts matching 'test_*.py' pattern."""
    test_loader = TestLoader()
    test_result = TestResult()

    test_suite = test_loader.discover(test_directory, pattern='test_*.py')
    test_suite.run(result=test_result)


def load_coverage_report() -> Union[Any, FileNotFoundError]:
    """
    Tries to find index.html file in given directory.
    If found, open in browser.
    If not, raise error.
    """
    filepath = IND_DIR
    if os.path.exists(filepath):
        new = 2  # open in a new tab, if possible
        webbrowser.open('file://' + filepath, new=new)
    else:
        raise FileNotFoundError
    return None


def prepare_path(global_path: str, make_dir: bool = False) -> Union[str, NotADirectoryError]:
    """Returns relative file directory."""
    # Create directory
    if make_dir and not os.path.exists(global_path):  # Creates folder if folder does not exist
        os.makedirs(global_path)
    return global_path
# endregion


if __name__ == '__main__':
    # run_tests(test_directory=TEST_DIR)
    write_coverage_report()
    load_coverage_report()
