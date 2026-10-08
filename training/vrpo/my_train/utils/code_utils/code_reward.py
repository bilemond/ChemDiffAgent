from taco import run_test as taco_run_test
import ast
import json
import multiprocessing
import re
from multiprocessing import Manager
from typing import Any


def extract_code_from_model(model_response: str):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()

def check_correctness(tests: list[dict[str, str]] | dict[str, list[str]], code: str, test_fn, timeout_per_test: int = 12, max_tests: int = 15) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all test cases within a timeout period.

    Args:
        tests: Test cases in either list of dictionaries or dictionary of lists format
        code: Generated code to test
        test_fn: Function to run tests
        timeout: Maximum execution time in seconds before killing process

    Returns:
        tuple: (bool, dict) where:
            - bool: True if all tests pass, False otherwise
            - dict: Detailed test results with test cases and pass/fail status
    """
    manager = Manager()
    test_results = manager.list()

    def evaluate_code(tests, generation, debug, test_results, test_fn):
        """Helper function to run tests in separate process."""
        try:
            test_results.append(test_fn(tests, test=generation, debug=debug, timeout=timeout_per_test))
        except Exception as e:
            print(f"Error in evaluate_code: {e}")

    original_tests = tests
    if isinstance(tests, list):
        list_tests = tests
        total_tests = len(list_tests)
        if total_tests > max_tests:
            # Sort indices by test input length and take the max_tests longest ones
            selected_indices = sorted(range(total_tests), key=lambda i: len(list_tests[i]["input"]), reverse=True)[:max_tests]
            tests = [list_tests[i] for i in selected_indices]
        num_tests = len(tests)
    else:
        dict_tests = tests
        total_tests = len(dict_tests["inputs"])
        if total_tests > max_tests:
            # Select the tests with the longest input length.
            selected_indices = sorted(range(total_tests), key=lambda i: len(dict_tests["inputs"][i]), reverse=True)[:max_tests]
            # Create a new dict with only the selected test cases
            selected_tests: dict[str, list[str]] = {"inputs": [dict_tests["inputs"][i] for i in selected_indices], "outputs": [dict_tests["outputs"][i] for i in selected_indices]}
            tests = selected_tests
        num_tests = len(tests["inputs"])

    process = multiprocessing.Process(target=evaluate_code, args=(tests, code, False, test_results, test_fn))
    process.start()
    process.join()

    if process.is_alive():
        process.kill()
    test_results_list = list(test_results)

    detailed_results: dict[str, Any] = {"all_passed": False, "test_results": [], "total_tests": num_tests, "passed_tests": 0}

    if len(test_results_list) == 0:
        return False, detailed_results

    test_results_data = test_results_list[0]
    passed_results = [r == True for r in test_results_data]

    # Create detailed test results
    test_results_list_typed: list[dict[str, Any]] = detailed_results["test_results"]
    if isinstance(original_tests, list):
        assert isinstance(tests, list)
        for i, (test, result) in enumerate(zip(tests, passed_results, strict=False)):
            test_results_list_typed.append({"input": test.get("input", ""), "expected": test.get("output", ""), "passed": result})
    else:
        assert isinstance(tests, dict)
        for i, (inp, out, result) in enumerate(zip(tests["inputs"], tests["outputs"], passed_results, strict=False)):
            test_results_list_typed.append({"input": inp, "expected": out, "passed": result})

    detailed_results["passed_tests"] = sum(passed_results)
    detailed_results["all_passed"] = all(passed_results)

    return all(passed_results), detailed_results

# https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems
def primeintellect_check_correctness(tests, code, use_tci=False):
    if isinstance(tests, str):
        try:
            tests = ast.literal_eval(tests)
            assert isinstance(tests, dict)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing string: {e}")
            return False, {"all_passed": False, "error": str(e)}

    assert len(tests) >= 1, "PrimeIntellect needs at least one test case"
    # Convert the tests to the format expected by the taco_run_test function
    inputs = [t["input"] for t in tests]
    outputs = [t["output"] for t in tests]
    fn_name = tests[0].get("fn_name", None)
    tests_formatted = {
        "inputs": inputs,
        "outputs": outputs,
    }
    if fn_name:
        tests_formatted["fn_name"] = fn_name

    return check_correctness(tests_formatted, code, taco_run_test)