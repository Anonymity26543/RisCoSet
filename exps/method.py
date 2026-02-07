import os
import sys
import numpy as np
from typing import List, Callable, Dict, Union, Any
import traceback
import argparse

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)

from astparser import parse_results
from astparser import ast_helper
from utils import utils
import optimize
from utils.io_tools import Tools
from utils.dataloader import DataLoader

# Global Variable
match_node_num = 0

PATH_DATASET = f"{ROOT_DIR}/data/datasets"
PATH_TO_OUTPUT = f"{ROOT_DIR}/output"

# datasets
datasets = ["HumanEval"]
# datasets = ["MBPP"]
# datasets = ["APPS"]


def retrieve_results(
    output: Any, fn: Callable, directories: List[str], path: str
) -> Any:

    for directory in directories:
        for file in [f for f in os.listdir(f"{path}/{directory}")]:
            output = fn(f"{path}/{directory}/{file}", output)
    return output


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
        
        
def code_normalizaiton(logprobs: dict):
    # first token is "def"
    # try:
    #     def_index = logprobs['tokens'].index("def")
    #     logprobs['tokens'] = logprobs['tokens'][def_index:]
    #     logprobs['token_logprobs'] = logprobs['token_logprobs'][def_index:]
    # except: 
    #     print("Error: not find def!")
    #     exit(0)
        
    # lstrip
    while True:
        if logprobs['tokens'][0].lstrip() == "":
            logprobs['tokens'] = logprobs['tokens'][1:]
            logprobs['token_logprobs'] = logprobs['token_logprobs'][1:]
            continue
        else:
            logprobs['tokens'][0] = logprobs['tokens'][0].lstrip()
            break
            
    # rstrip
    while True:
        if logprobs['tokens'][-1].rstrip() == "":
            logprobs['tokens'] = logprobs['tokens'][:-1]
            logprobs['token_logprobs'] = logprobs['token_logprobs'][:-1]
            continue
        else:
            logprobs['tokens'][-1] = logprobs['tokens'][-1].rstrip()
            break     
        
    # use '' instead of ""
    logprobs["tokens"] = [
        tok.replace("\"", "'") for tok in logprobs["tokens"]
    ]
        
    # only single '\n'
    before_length = len(logprobs['tokens'])
    after_length = before_length
    bias = 0
    for i in range(logprobs['tokens'].index("def"), before_length):
        i = i + bias
        if logprobs['tokens'][i] == '\n' and '\n' in logprobs['tokens'][i + 1:]:
            next_index = logprobs['tokens'][i + 1:].index('\n') + i + 1
            if "".join(logprobs['tokens'][i:next_index + 1]).strip() == "":
                logprobs['tokens'] = logprobs['tokens'][:i + 1] + logprobs['tokens'][next_index + 1:]
                logprobs['token_logprobs'] = logprobs['token_logprobs'][:i + 1] + logprobs['token_logprobs'][next_index + 1:]
                after_length = after_length - next_index + i
                bias = bias - 1
                continue
                
        if i + 1 > after_length - 1:
            break

    return logprobs


def is_subtree(
    target_root_code: str,
    target: ast_helper.Node,
    pruned_root_code: str,
    pruned: ast_helper.Node,
    is_pruned_root: bool,
) -> Dict[str, Union[bool, str]]:

    # check if pruned tree is subtree of target tree
    def get_string_intervals(code, n):
        return "#".join([code[t[0] : t[1]] for t in n.intervals])
    
    if is_pruned_root and pruned.deleted:
        return {"eval": True, "reason": "Pruned is empty"}

    if pruned is None:
        return {"eval": True, "reason": "Pruned is None"}
    
    curr_node_same = get_string_intervals(
        target_root_code, target
    ) == get_string_intervals(pruned_root_code, pruned)
    
    if curr_node_same:

        def map_string_interval_to_ind(root_code, children):
            map_interval_to_ind = {}
            for i in range(len(children)):
                if not children[i].deleted:
                    t = children[i].intervals
                    map_interval_to_ind[
                        get_string_intervals(root_code, children[i])
                    ] = i
            return map_interval_to_ind

        target_children_map_interval_to_ind = map_string_interval_to_ind(
            target_root_code, target.children
        )
        pruned_children_map_interval_to_ind = map_string_interval_to_ind(
            pruned_root_code, pruned.children
        )
        # check if all pruned children are in target children
        for pruned_child_str in pruned_children_map_interval_to_ind:
            if pruned_child_str not in target_children_map_interval_to_ind:
                return {
                    "eval": False,
                    "reason": f"Pred child ({pruned_child_str}) not in target_children_set ({target_children_map_interval_to_ind.keys()})",
                }
        # check if pruned child node is subtree of relevant target children node
        for target_child_str in target_children_map_interval_to_ind:
            if target_child_str in pruned_children_map_interval_to_ind:
                eval = is_subtree(
                    target_root_code,
                    target.children[
                        target_children_map_interval_to_ind[target_child_str]
                    ],
                    pruned_root_code,
                    pruned.children[
                        pruned_children_map_interval_to_ind[target_child_str]
                    ],
                    False,
                )
                if not eval["eval"]:
                    return eval
        return {"eval": True}
    else:
        return {
            "eval": False,
            "reason": "Curr nodes are not the same: "
            + str([target_root_code[t[0] : t[1]] for t in target.intervals])
            + " vs "
            + str([pruned_root_code[t[0] : t[1]] for t in pruned.intervals]),
        }


def get_new_targets(
        targets_dict: dict,
        solution_num = 1, 
    ):
    if solution_num == 0:
        return targets_dict
    
    results = utils.read_json(
        f"{ROOT_DIR}/output/results/{datasets[0]}/verification_80_30.json"
        )["output"]

    for task_id in targets_dict:
        solutions = results[task_id][:solution_num]
        for solution in solutions:
            if solution["passed"]:
                targets_dict[task_id].append(solution["raw_program"])
                
    return targets_dict


if __name__ == "__main__":
    isprint = True
    issave = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", dest="m", type=int, default=1)
    parser.add_argument("--s", dest="s", type=int, default=20)
    args = parser.parse_args()
    

    # targets
    targets_dict = {}
    targets = retrieve_results(
        [],
        lambda x, output: output + Tools.load_jsonl(x),
        datasets,
        PATH_DATASET,
    )
    for target in targets:
        if target["task_id"] not in targets_dict:
            function_def = target["prompt"][target["prompt"].find("def"):]
            function_def = function_def[:function_def.find("\n") + 1]
            targets_dict[target["task_id"]] = [function_def + target["canonical_solution"]]
        else:
            task_id = target["task_id"]
            print(f"task_id repeat: {task_id}")
            exit(0)


    # get new targets via executing solutions
    new_targets_dict = get_new_targets(
        targets_dict, 
        args.s,
    )
    
    # results
    results = retrieve_results(
        [],
        lambda x, output: output + Tools.load_jsonl(x),
        datasets,
        PATH_TO_OUTPUT,
    )

    Lambda = np.linspace(1e-3, 1 - 1e-3, 50)
    max_costs = [-np.log(x) for x in Lambda]
    print("max_costs: ", max_costs)

    output_data = []
    cnt_valid = 0
    cnt_match = 0
    frac_included_list = []
    ppl_match_list = []
    ppl_mismatch_list = []
    # print(results)
    print("Number of problems: ", len(results))
    # results = [results[args.dataind]] if args.dataind >= 0 else results

    for i, sample in enumerate(results):
        print(f"[{i}/{len(results)-1}]{'-'*10}", flush=True)
        try:
            sample["logprobs"] = filter_annotation_token(sample["logprobs"])
            sample["logprobs"] = code_normalizaiton(sample["logprobs"])
            sample["completion"] = "".join(sample["logprobs"]["tokens"])
            pred_str = sample["completion"]
            target_str_list = new_targets_dict[sample["task_id"]]
        except:
            traceback.print_exc()
            continue


        target_tree_list = []
        target_str_list_2 = []
        for target_str in target_str_list:
            target_tree = parse_results.code_to_final_ast(target_str.strip())
            if target_tree != None:
                target_tree_list.append(target_tree)
                target_str_list_2.append(target_str)
            else:
                print('error label!')
        print("num of labels: ", len(target_tree_list))
        
        pred_tree = parse_results.code_to_final_ast(pred_str.strip())
        if len(target_tree_list) == 0 or pred_tree == None:
            continue
        
        try:
            parse_results.add_probability_to_nodes(
                pred_tree, sample, False
            )

        except:
            traceback.print_exc()
            continue
        
        pruned_tree_data = optimize.create_tree_from_optimization_result_lst(
            pred_tree, args.m, max_costs
        )
        if pruned_tree_data == None:
            print('Optimization failure!')
            continue
        
        for j, optimize_output in enumerate(pruned_tree_data):
            save_data = {}
            frac_included_list.append(optimize_output["frac_included"])
            try:
                if isprint:
                    cur_target_tree = None
                    cur_target_str = None
                    for k, target_tree in enumerate(target_tree_list):
                        if target_tree == None:
                            continue
                        subtree_res = is_subtree(
                            target_tree.code,
                            target_tree,
                            pred_tree.code,
                            optimize_output["entire_tree_with_deleted"],
                            True,
                        )

                        cur_target_tree = target_tree
                        cur_target_str = target_str_list_2[k]
                        if subtree_res["eval"]:
                            cnt_match += 1
                            break
                        
                save_data["task_id"] = sample["task_id"]
                save_data["pred_in_target"] = subtree_res
                save_data["pred_str"] = pred_str.strip()
                save_data["target_str"] = cur_target_str.strip()
                save_data["cost"] = max_costs[j]
                save_data["Lambda"] = Lambda[j]
                save_data["data_ind"] = i
                save_data["Lambda_ind"] = j
                save_data["output"] = optimize_output
                save_data["output"]["pruned_root"] = save_data["output"][
                    "pruned_root"
                ]
                if save_data["output"]["pruned_root"] != None:
                    save_data["output"]["pruned_root"] = save_data["output"]["pruned_root"].toJSON()
                save_data["output"]["entire_tree_with_deleted"] = save_data["output"][
                    "entire_tree_with_deleted"
                ].toJSON()
                if "check" in save_data["output"]:
                    save_data["output"]["check"] = str(save_data["output"]["check"])
                output_data.append(save_data)
                cnt_valid += 1
            except:
                traceback.print_exc()
                continue
        

    print(cnt_match, cnt_valid, 1.0 * cnt_match / cnt_valid)
    print(1.0 - np.mean(frac_included_list))

    if issave:
        utils.write_json(
            f"{ROOT_DIR}/output/results/{datasets[0]}/output__m_{args.m}__s_{args.s}.json",
            {"output": output_data},
        )

            
            

