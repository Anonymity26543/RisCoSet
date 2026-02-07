import os
from utils.io_tools import Tools, cache
from utils.parser import Parser, FunctionCallVisitor
from utils.prompt import *
from typing import List, Dict, TypedDict
from collections import defaultdict, Counter
from config.typing import *

import random

from tqdm import tqdm
import logging

DATAPATH_DIR = "../data/datasets"
DEBUG_FILE = "../output/debug/debug.txt"



def filter_annotation_token(logprobs: dict):
    for i in range(len(logprobs['tokens'])):
        if '\"\"\"' in logprobs['tokens'][i]:
            logprobs['tokens'][i] = '\"\"\"'
        elif "'''" in logprobs['tokens'][i]:
            logprobs['tokens'][i] = "'''"
        elif logprobs['tokens'][i].strip() == '#':
            logprobs['tokens'][i] = '#'            
        else:
            continue
        
    while True:
        try:
            index1 = logprobs['tokens'].index('\"\"\"')
            index2 = logprobs['tokens'][index1 + 1:].index('\"\"\"') + index1 + 1
            logprobs['tokens'] = logprobs['tokens'][:index1] + logprobs['tokens'][index2 + 1:]
            logprobs['token_logprobs'] = logprobs['token_logprobs'][:index1] + logprobs['token_logprobs'][index2 + 1:]
            continue
        except:
            break
        
    while True:
        try:
            index1 = logprobs['tokens'].index("'''")
            index2 = logprobs['tokens'][index1 + 1:].index("'''") + index1 + 1
            logprobs['tokens'] = logprobs['tokens'][:index1] + logprobs['tokens'][index2 + 1:]
            logprobs['token_logprobs'] = logprobs['token_logprobs'][:index1] + logprobs['token_logprobs'][index2 + 1:]
            continue
        except:
            break
        
        
    while True:
        try:
            index1 = logprobs['tokens'].index('#')
            if '\n' in logprobs['tokens'][index1 + 1:]:
                index2 = logprobs['tokens'][index1 + 1:].index('\n') + index1 + 1
            else:
                logprobs['tokens'] = logprobs['tokens'][:index1]
                logprobs['token_logprobs'] = logprobs['token_logprobs'][:index1]
                return logprobs
            logprobs['tokens'] = logprobs['tokens'][:index1] + logprobs['tokens'][index2:]
            logprobs['token_logprobs'] = logprobs['token_logprobs'][:index1] + logprobs['token_logprobs'][index2:]
            continue
        except:
            return logprobs


class DataLoader:

    @staticmethod
    def get_absolute_dataset_path(dataset: str) -> str:
        return os.path.join(DATAPATH_DIR, f"{dataset}/{dataset}.jsonl")

    @staticmethod
    def load_prompt(dataset: str, prompt_type: str) -> List[Dict]:
        problems = DataLoader.load_problems(
            os.path.join(DATAPATH_DIR, f"{dataset}/{dataset}.jsonl")
        )
        if prompt_type == "solution":
            return [
                {
                    "task_id": problem["task_id"],
                    "prompt": SolutionPrompt(problem),
                    "entry_point": problem["entry_point"],
                }
                for problem in problems.values()
            ]

        elif prompt_type == "testcase":
            return [
                {
                    "task_id": problem["task_id"],
                    "prompt": TestcasePrompt(problem),
                }
                for problem in problems.values()
            ]
        elif prompt_type == "inputgen":
            return [
                {
                    "task_id": problem["task_id"],
                    "prompt": InputgenPrompt(problem),
                }
                for problem in problems.values()
            ]
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")

    @staticmethod
    def load_problems(problems_path: str) -> Dict[str, Problem]:
        # problems = {problem['task_id']: problem for problem in Tools.load_jsonl(problems_path)}
        problems = {}
        for problem in Tools.load_jsonl(problems_path):
            problem["test"] = "def" + problem["test"].split("def")[1]
            problem["description"] = problem["prompt"]
            problems[problem["task_id"]] = Problem(**problem)
        return problems

    @staticmethod
    # @cache
    def load_solutions(
        problems_path: str, solutions_path: str, solutions_num: int, verbal=False
    ) -> Dict[str, List]:
        """
        Load solutions jsonl file from SOLUTIONS_PATH and problems jsonl file from PROBLEMS_PATH, and then package them
        into a list of dictionaries, corresponding to solution samples will be being executed.
        """
        print("Loading solutions...")
        problems = {
            problem["task_id"]: problem for problem in Tools.load_jsonl(problems_path)
        }
        packed_solutions = defaultdict(list)
        solution_id = Counter()
        for solution in tqdm(Tools.load_jsonl(solutions_path)):
            task_id = solution["task_id"]
            if solution_id[task_id] >= solutions_num > 0:  # solutions_num limit
                continue
            
            solution["logprobs"] = filter_annotation_token(solution["logprobs"])
            solution["completion"] = "".join(solution["logprobs"]["tokens"])
            solution["entry_point"] = problems[task_id]["entry_point"]
            packed_solutions[task_id].append(
                {
                    "task_id": task_id,
                    "solution_id": solution_id[task_id],
                    "solution": Parser.extract_normal_solution(
                        solution["completion"].replace("\t", "    ").split('\n', 1)[-1]
                        )[0],
                    "raw_solution": solution["completion"],
                    # 'cumulative_logprob': solution['cumulative_logprob'],
                    # "logprobs_len": solution["logprobs_len"],
                    # "mean_logprob": solution["mean_logprob"],
                    "ppl": solution["ppl"],
                    # "verbal_confidence": (
                        # Parser.extract_verbal_confidence(solution["completion"])
                        # if verbal
                        # else 0
                    # ),
                }
            )
            solution_id[task_id] += 1

        # change this dict to list format like :
        # [ { task_id: HumanEval/01, solutions : [{solution1, solution2, ...}],
        #   { task_id: HumanEval/02, solutions : [...]},
        # ...]
        processed_solutions = []
        for task_id, solution_list in packed_solutions.items():
            processed_solution = dict()
            processed_solution["task_id"] = task_id
            processed_solution["solutions"] = solution_list
            processed_solutions.append(processed_solution)

        return processed_solutions

    @staticmethod
    # @cache
    def load_solutions_with_batch(
        problems_path: str,
        solutions_path: str,
        solutions_num: int,
        start_index: int,
        end_index: int,
        verbal=False,
    ) -> Dict[str, List]:
        """
        Load solutions jsonl file from SOLUTIONS_PATH and problems jsonl file from PROBLEMS_PATH, and then package them
        into a list of dictionaries, corresponding to solution samples will be being executed.
        """
        print(f"Loading solutions... From index {start_index} to {end_index}")
        problems = {
            problem["task_id"]: problem
            for problem in Tools.load_jsonl(problems_path)[start_index:end_index]
        }
        packed_solutions = defaultdict(list)
        solution_id = Counter()
        batched_raw_solutions = Tools.load_jsonl(solutions_path)[start_index:end_index]
        for solution in tqdm(batched_raw_solutions):
            task_id = solution["task_id"]
            if solution_id[task_id] >= solutions_num > 0:  # solutions_num limit
                continue
            solution["entry_point"] = problems[task_id]["entry_point"]
            packed_solutions[task_id].append(
                {
                    "task_id": task_id,
                    "solution_id": solution_id[task_id],
                    "solution": Parser.extract_normal_solution(solution["completion"])[
                        0
                    ],
                    # 'cumulative_logprob': solution['cumulative_logprob'],
                    "logprobs_len": solution["logprobs_len"],
                    "mean_logprob": solution["mean_logprob"],
                    "ppl": solution["ppl"],
                    "verbal_confidence": (
                        Parser.extract_verbal_confidence(solution["completion"])
                        if verbal
                        else 0
                    ),
                }
            )
            solution_id[task_id] += 1

        # change this dict to list format like :
        # [ { task_id: HumanEval/01, solutions : [{solution1, solution2, ...}],
        #   { task_id: HumanEval/02, solutions : [...]},
        # ...]
        processed_solutions = []
        for task_id, solution_list in packed_solutions.items():
            processed_solution = dict()
            processed_solution["task_id"] = task_id
            processed_solution["solutions"] = solution_list
            processed_solutions.append(processed_solution)

        return processed_solutions

    @staticmethod
    def load_processed_solution(
        processed_solution_path: str, solution_num: int
    ) -> List[Dict]:

        processed_solutions = []
        for item in tqdm(Tools.load_jsonl(processed_solution_path)):
            processed_solutions = processed_solutions + item["solutions"]
        return processed_solutions

    @staticmethod
    # @cache
    def load_testcases(
        problems_path: str,
        testcases_path: str,
        inputgen_path: str,
        testcase_type: str,
        testcases_num: int,
    ) -> Dict[str, List]:
        """
        Load testcases jsonl file from TESTCASES_PATH and problem jsonl file from PROBLEMS_PATH, and then extract
        testcases of each problem task, testcase type is specified by TESTCASE_TYPE.
        """
        print("Loading test cases...")
        problems = {
            problem["task_id"]: problem for problem in Tools.load_jsonl(problems_path)
        }
        testcases = Tools.load_jsonl(testcases_path)

        raw_testcases = defaultdict(list)
        raw_testcases_input = defaultdict(list)
        packed_testcases = defaultdict(list)

        visitor = FunctionCallVisitor()
        testcases_set = set()

        non_duplicate_testcases = defaultdict(set)

        for testcase_json in tqdm(testcases):
            task_id = testcase_json["task_id"]
            problem = problems[task_id]

            if (
                testcase_json["completion"] in testcases_set
            ):  # remove duplicated test cases
                continue
            testcases_set.add(testcase_json["completion"])

            if testcase_type == "io":
                testcase_list = Parser.extract_testcase_with_io(
                    testcase_json["completion"], problem["entry_point"]
                )
            else:
                testcase_list = Parser.extract_testcase(
                    testcase_json["completion"], problem["entry_point"]
                )

            for testcase in testcase_list:
                hashed_io = str(testcase["input"]) + " : " + str(testcase["output"])
                if hashed_io in non_duplicate_testcases[task_id]:
                    continue
                non_duplicate_testcases[task_id].add(hashed_io)
                raw_testcases[task_id].append(testcase)

            if testcase_type == "input":
                for testcase in raw_testcases[task_id]:
                    test_input = visitor(problem["entry_point"].strip(), testcase)
                    if test_input:
                        raw_testcases_input[task_id].append(test_input)
        print(f"testcases num: {len(raw_testcases)}, problems num: {len(problems)}")
        assert len(raw_testcases) == len(problems), "Incomplete testcases!"

        if testcase_type == "assertion":
            packed_testcases = raw_testcases
        elif testcase_type == "input":
            packed_testcases = raw_testcases_input
        elif testcase_type == "input_gen":
            # results = execute_inputgen_prog(Tools.load_jsonl(inputgen_path))
            result = 0  # TODO : evaluate
            for task_id in problems:
                problem = problems[task_id]
                if task_id in results:
                    packed_testcases[task_id] = [
                        f"{problem['entry_point']}(*{str(test_param)})"
                        for test_param in results[task_id]
                    ]
                else:
                    packed_testcases[task_id] = raw_testcases[task_id]

                if testcase_type == "mix":
                    packed_testcases[task_id] += raw_testcases_input[task_id]
        elif testcase_type == "io":
            testcase_id = Counter()
            for task_id, testcase_per_task in raw_testcases.items():

                if len(testcase_per_task) == 0:
                    packed_testcases[task_id] = []
                    continue

                for testcase_sample in testcase_per_task:
                    packed_testcases[task_id].append(
                        {
                            "task_id": task_id,
                            "testcase_id": testcase_id[task_id],
                            "raw": testcase_sample["raw"],
                            "input": testcase_sample["input"],
                            "output": testcase_sample["output"],
                        }
                    )
                    testcase_id[task_id] += 1

        packed_testcases = {
            key: (
                random.sample(value, testcases_num)
                if 0 < testcases_num <= len(value)
                else value
            )
            for key, value in packed_testcases.items()
        }
        # Calculate the average length of testcases.
        # s = 0
        # for testcases in packed_testcases.values():
        #     s += len(testcases)
        # print(f"Average length of testcases: {s / len(packed_testcases)}")
        # print(f"Minimum count of testcases: {min([len(tc) for tc in packed_testcases.values()])}")
        # print(len(packed_testcases))
        return packed_testcases

    @staticmethod
    # @cache
    def load_testcases_with_batch(
        problems_path: str,
        testcases_path: str,
        inputgen_path: str,
        testcase_type: str,
        testcases_num: int,
        start_index: int,
        end_index: int,
    ) -> Dict[str, List]:
        """
        Load testcases jsonl file from TESTCASES_PATH and problem jsonl file from PROBLEMS_PATH, and then extract
        testcases of each problem task, testcase type is specified by TESTCASE_TYPE.
        """

        print(f"Loading test cases..., loading index : {start_index} to {end_index} ")
        problems = {
            problem["task_id"]: problem
            for problem in Tools.load_jsonl(problems_path)[start_index:end_index]
        }
        testcases = Tools.load_jsonl(testcases_path)[start_index:end_index]

        raw_testcases = defaultdict(list)
        raw_testcases_input = defaultdict(list)
        packed_testcases = defaultdict(list)

        visitor = FunctionCallVisitor()
        testcases_set = set()
        for testcase_json in tqdm(testcases):
            task_id = testcase_json["task_id"]
            problem = problems[task_id]

            if (
                testcase_json["completion"] in testcases_set
            ):  # remove duplicated test cases
                continue
            testcases_set.add(testcase_json["completion"])

            if testcase_type == "io":
                testcase_list = Parser.extract_testcase_with_io(
                    testcase_json["completion"], problem["entry_point"]
                )
            else:
                testcase_list = Parser.extract_testcase(
                    testcase_json["completion"], problem["entry_point"]
                )
            raw_testcases[task_id].extend(testcase_list)
            if testcase_type == "input":
                for testcase in raw_testcases[task_id]:
                    test_input = visitor(problem["entry_point"].strip(), testcase)
                    if test_input:
                        raw_testcases_input[task_id].append(test_input)

        assert len(raw_testcases) == len(problems), "Incomplete testcases!"

        if testcase_type == "assertion":
            packed_testcases = raw_testcases
        elif testcase_type == "input":
            packed_testcases = raw_testcases_input
        elif testcase_type == "input_gen":
            # results = execute_inputgen_prog(Tools.load_jsonl(inputgen_path))
            result = 0  # TODO : evaluate
            for task_id in problems:
                problem = problems[task_id]
                if task_id in results:
                    packed_testcases[task_id] = [
                        f"{problem['entry_point']}(*{str(test_param)})"
                        for test_param in results[task_id]
                    ]
                else:
                    packed_testcases[task_id] = raw_testcases[task_id]

                if testcase_type == "mix":
                    packed_testcases[task_id] += raw_testcases_input[task_id]
        elif testcase_type == "io":
            testcase_id = Counter()
            for task_id, testcase_per_task in raw_testcases.items():

                if len(testcase_per_task) == 0:
                    packed_testcases[task_id] = []
                    continue

                for testcase_sample in testcase_per_task:
                    packed_testcases[task_id].append(
                        {
                            "task_id": task_id,
                            "testcase_id": testcase_id[task_id],
                            "raw": testcase_sample["raw"],
                            "input": testcase_sample["input"],
                            "output": testcase_sample["output"],
                        }
                    )
                    testcase_id[task_id] += 1

        packed_testcases = {
            key: (
                random.sample(value, testcases_num)
                if 0 < testcases_num <= len(value)
                else value
            )
            for key, value in packed_testcases.items()
        }
        # Calculate the average length of testcases.
        # s = 0
        # for testcases in packed_testcases.values():
        #     s += len(testcases)
        # print(f"Average length of testcases: {s / len(packed_testcases)}")
        # print(f"Minimum count of testcases: {min([len(tc) for tc in packed_testcases.values()])}")
        # print(len(packed_testcases))
        return packed_testcases

    @staticmethod
    def load_clusters(clusters_path: str) -> List[dict]:
        """
        Load clusters jsonl file from CLUSTERS_PATH, and then package them
        into a list of dictionaries, corresponding to cluster samples.
        return type:
            "task_id": str
            "clusters": list[Cluster]
        """
        rawdata = Tools.load_jsonl(clusters_path)
        processed_clusters = []
        for item in rawdata:
            entry = dict()
            entry["task_id"] = item["task_id"]
            entry["clusters"] = []
            for cluster in item["result"]:
                entry["clusters"].append(
                    Cluster(
                        task_id=item["task_id"],
                        exec_result=None,
                        solutions_list=cluster["solution_ids"],
                        total=cluster["total"],
                        score=0,
                        passed=cluster["correct"],
                    )
                )
            processed_clusters.append(entry)
        return processed_clusters

    @staticmethod
    def load_exec_result(exec_result_path: str) -> List[ExecResult]:
        """
        Load exec result jsonl file from EXEC_RESULT_PATH, and then package them
        into a list of dictionaries, corresponding to exec result samples.
        """
        # logger = logging.getLogger(__name__)
        rawdata = Tools.load_jsonl(exec_result_path)
        exec_result = []
        for entry in rawdata:
            exec_result_list = []
            for exec_result_per_testcase in entry["exec_result"]:
                normalized_output = "<SE>"
                try:
                    normalized_output = normalize_result(
                        exec_result_per_testcase["output"]
                    )
                except Exception as e:
                    # logger.error(f"Error normalizing output: {str(e)}")
                    # logger.error(f"Error File: {exec_result_path.split('/')[-1]}")
                    # logger.error(
                    #     f"testcase_id: {exec_result_per_testcase['testcase_id']}"
                    # )
                    pass

                exec_result_list.append(
                    ExecResultPerTestcase(
                        testcase_id=exec_result_per_testcase["testcase_id"],
                        result=exec_result_per_testcase["result"],
                        output=normalized_output,
                    )
                )
            exec_result.append(
                ExecResult(
                    solution_id=entry["solution_id"],
                    passed=entry["ground_truth"],
                    ppl=entry["ppl"],
                    exec_result=exec_result_list,
                )
            )
        return exec_result
