import os
import sys

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)

from utils import utils
from execution import check_correctness
from utils.dataloader import DataLoader


PATH_DATASET = f"{ROOT_DIR}/data/datasets"
PATH_TO_OUTPUT = f"{ROOT_DIR}/output"

# datasets
datasets = ["HumanEval"]
# datasets = ["MBPP"]
# datasets = ["APPS"]

def execution(
    problem_path: str,
    solution_path: str,
    solution_num: int,
    worker_num: int,
):
    problems = DataLoader.load_problems(problem_path)
    solutions = DataLoader.load_solutions(problem_path, solution_path, solution_num)
    results = check_correctness(problems, solutions, num_workers=worker_num)
    return results


def get_new_targets(
        problem_path: str, 
        solution_path: str, 
        solution_num = 1, 
        worker_num = 10,
    ):
    
    filename = []
    for file in [f for f in os.listdir(f"{solution_path}")]:
        filename.append(file)
        
    results = execution(problem_path, solution_path + f"/{filename[0]}", solution_num, worker_num)

    utils.write_json(
        f"{ROOT_DIR}/output/results/{datasets[0]}/verification_{solution_num}_{worker_num}.json",
        {"output": results},
    )


if __name__ == "__main__":
    # get new targets via executing solutions
    get_new_targets( 
        PATH_DATASET + f"/{datasets[0]}/{datasets[0]}.jsonl",
        PATH_TO_OUTPUT + f"/{datasets[0]}-sampling",
        80, 30,
    )


    # data = utils.read_json(
    #     f"{ROOT_DIR}/output/results/{datasets[0]}/verification_80_30.json"
    # )["output"]

    # n = 0
    # ac = 0
    # for task_id in data:
    #     flag = 0
    #     for solution in data[task_id]:
    #         n = n + 1
    #         if solution["passed"]:
    #             ac = ac + 1
    #         else:
    #             flag = 1
    #     # if flag == 0:
    #     #     print(task_id)

                
    # print(ac, n, ac/n)



