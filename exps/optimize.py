from collections import deque
from z3 import *
import os
import sys
import numpy as np
from typing import List, Tuple, Dict, Union, Any

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)

from astparser.parse_results import *
from utils.io_tools import Tools
import astparser.ast_helper as ast_helper

# Global Variable
pruned_indices = []


def add_tree_constraints(
    o: z3.z3.Optimize, tree: ast_helper.Node, cost_id: str = "", m: int = -1
) -> Tuple[List[float], List[Bool]]:
    ordered_probabilities = []
    indicator_variables = []
    map_node_to_indicator = {}
    # traverse tree and create ordered names of nodes and probabilities corresponding to that node
    # create boolean variables for inclusion in tree
    node_number = 0
    q = deque()
    q.append(tree)
    while len(q) > 0:
        curr_node = q.popleft()
        curr_indicator = Bool(
            curr_node.code + "::" + str(node_number)
            if cost_id == ""
            else str(cost_id) + "::" + curr_node.code + "::" + str(node_number)
        )
        # ignore nodes in tree without prob
        if curr_node.nll != -1 and not (np.isnan(curr_node.nll)):
            indicator_variables.append(curr_indicator)
            map_node_to_indicator[curr_node] = curr_indicator
            # make sure negative log likelihood is between 10 and 0
            ordered_probabilities.append(min(10, max(curr_node.nll, 0)))
        for c in curr_node.children:
            q.append(c)
        node_number += 1

    # add constraints (linear time)
    q = deque()
    q.append((tree, None))
    while len(q) > 0:
        curr_node, parent_indicator = q.popleft()
        curr_indicator_variable = (
            map_node_to_indicator[curr_node]
            if curr_node in map_node_to_indicator
            else parent_indicator
        )
        for c in curr_node.children:
            if c in map_node_to_indicator:
                child_indicator_variable = map_node_to_indicator[c]
                o.add(
                    Implies(child_indicator_variable, curr_indicator_variable)
                )  # this is the converse of the child removal constraint
                # print("implication: child: " + c.code + " ->  parent: " + curr_node.code)
            else:
                for grand_child in c.children:
                    if grand_child in map_node_to_indicator:
                        child_indicator_variable = map_node_to_indicator[grand_child]
                        o.add(
                            Implies(child_indicator_variable, curr_indicator_variable)
                        )
                        # print("implication: grandchild: " + grand_child.code + " ->  parent: " + curr_node.code)
            q.append((c, curr_indicator_variable))

    assert len(ordered_probabilities) == len(indicator_variables)
    # make sure all float
    ordered_probabilities = [
        p if type(p) == float else p - 1e-7 for p in ordered_probabilities
    ]

    if m != -1:
        # print("M=", m)
        HOLE_FLOAT = 0.99999
        # iterate through nodes and add constraint for number of holes
        sum_holes = 0
        for node in map_node_to_indicator:
            # print("parent|", node.code)
            for c in node.children:
                pass_down_children = [c]
                for child_node in pass_down_children:
                    if child_node not in map_node_to_indicator:
                        pass_down_children += child_node.children
                    else:
                        # print("\tchild|", child_node.code if child_node in map_node_to_indicator else "(not in map):" + child_node.code)
                        sum_holes += (
                            HOLE_FLOAT
                            * Not(map_node_to_indicator[child_node])
                            * map_node_to_indicator[node]
                            if node in map_node_to_indicator
                            and child_node in map_node_to_indicator
                            else 0
                        )
        o.add(sum_holes <= m)
    return ordered_probabilities, indicator_variables


def solve_optimization_lst(
    tree: ast_helper.Node, m: int, max_cost_threshold: List[float]
) -> Tuple[CheckSatResult, Model, int]:
    o = Optimize()
    assert all(
        [
            max_cost_threshold[i] >= max_cost_threshold[i + 1]
            for i in range(len(max_cost_threshold) - 1)
        ]
    )
    all_probabilities = []
    all_indicator_variables = []
    # add node removal constraints for each threshold
    for curr_max_cost_threshold in max_cost_threshold:
        probabilities, indicator_variables = add_tree_constraints(
            o, tree, cost_id=curr_max_cost_threshold, m=m
        )
        all_probabilities.append(probabilities)
        all_indicator_variables.append(indicator_variables)
        # add single tau level constraint
        o.add(
            sum( # constraint 9
                [
                    float(probabilities[i]) * indicator_variables[i]
                    for i in range(len(indicator_variables))
                ]
            )
            <= curr_max_cost_threshold
        )
    o.maximize(
        sum(
            [
                sum( # optimization goal
                    [
                        1.01 * all_indicator_variables[threshold_ind][indicator_ind]
                        for indicator_ind in range(
                            len(all_indicator_variables[threshold_ind])
                        )
                    ]
                )
                for threshold_ind in range(len(max_cost_threshold))
            ]
        )
    )
    # set timeout
    o.set("timeout", 24000) 
    result = o.check()
    model = o.model() if result == sat else None
    return result, model, len(all_indicator_variables[0])


def create_tree(
    tree: ast_helper.Node, check: CheckSatResult, tuples: List[Tuple[str, str]]
) -> Dict[
    str, Union[ast_helper.Node, CheckSatResult, List[Tuple[str, str]], float, float]
]:
    if str(check) == "unsat":
        return None
    else:
        map_node_name_to_include = {}
        for tup in tuples:
            if tup[0].count("::") >= 2:
                tup[0] = tup[0][tup[0].index("::") + 2 :]
            if len(tup) == 2:
                map_node_name_to_include[tup[0]] = tup[1] == "True"
        # print("Nodes Included", sum([map_node_name_to_include[key] for key in map_node_name_to_include]))

        included_nodes = 0
        total_nodes = 0
        error_of_tree = []
        pruned_root = ast_helper.Node("root_pruned")
        entire_tree_with_deleted = ast_helper.Node("root_entire")
        node_number = 0
        q = deque()
        q.append(
            {
                "pruned_tree_parent": pruned_root,
                "entire_tree_parent": entire_tree_with_deleted,
                "child": tree,
            }
        )
        while len(q) > 0:
            curr = q.popleft()
            curr_child = curr["child"]
            pruned_tree_curr_parent = curr["pruned_tree_parent"]
            entire_tree_curr_parent = curr["entire_tree_parent"]
            pruned_tree_curr_node = ast_helper.Node(curr_child.code)
            pruned_tree_curr_node.nll = curr_child.nll
            pruned_tree_curr_node.intervals = curr_child.intervals
            pruned_tree_curr_node.start = curr_child.start
            pruned_tree_curr_node.end = curr_child.end
            entire_tree_curr_node = ast_helper.Node(curr_child.code)
            entire_tree_curr_node.nll = curr_child.nll
            entire_tree_curr_node.intervals = curr_child.intervals
            entire_tree_curr_node.start = curr_child.start
            entire_tree_curr_node.end = curr_child.end

            total_nodes += 1
            name_of_curr = curr_child.code + "::" + str(node_number)
            pruned_tree_curr_node.colon_name = name_of_curr
            entire_tree_curr_node.colon_name = name_of_curr
            entire_tree_curr_parent.children.append(entire_tree_curr_node)
            # if adding to tree, create new node and add to children of parent
            # node in map if no prob associated with token
            if (
                name_of_curr not in map_node_name_to_include
                or map_node_name_to_include[name_of_curr]
            ):
                pruned_tree_curr_node.deleted = False
                entire_tree_curr_node.deleted = False
                pruned_tree_curr_parent.children.append(pruned_tree_curr_node)
                error_of_tree.append(
                    curr_child.nll if curr_child.nll not in [-1, np.nan] else 0
                )
                included_nodes += 1
            else:
                pruned_tree_curr_node.deleted = True
                entire_tree_curr_node.deleted = True

            # need to explore all children even if not including in tree to maintain node number count
            for c in curr_child.children:
                q.append(
                    {
                        "pruned_tree_parent": pruned_tree_curr_node,
                        "entire_tree_parent": entire_tree_curr_node,
                        "child": c,
                    }
                )
            node_number += 1
        return {
            "pruned_root": (
                pruned_root.children[0] if len(pruned_root.children) > 0 else None
            ),
            "entire_tree_with_deleted": (
                entire_tree_with_deleted.children[0]
                if len(entire_tree_with_deleted.children) > 0
                else None
            ),
            "map": map_node_name_to_include,
            "check": check,
            "tuples": tuples,
            "error_of_tree": sum(error_of_tree),
            "frac_included": included_nodes / total_nodes,
        }


def create_tree_from_optimization_result_lst(
    tree: ast_helper.Node, m: int, max_cost_threshold: List[float]
) -> List[
    Dict[
        str,
        Union[
            ast_helper.Node,
            CheckSatResult,
            Dict[str, bool],
            List[Tuple[str, str]],
            float,
            float,
        ],
    ]
]:

    check, model, _ = solve_optimization_lst(tree, m, max_cost_threshold)
    if model == None:
        return None
    pruned_tree_data = []
    # make list of tuples: (variable_id, variable)
    tuples = []
    for i in range(model.__len__()):
        key = model.__getitem__(i)
        value = model.get_interp(key)
        tuples.append((str(key), str(value)))

    # map tau setting to variables in the tau setting
    # print('tuples: ', tuples)
    map_tau_to_vars = {}
    for key, value in tuples:
        curr_tau = key.split("::")[0]
        if curr_tau not in map_tau_to_vars:
            map_tau_to_vars[curr_tau] = []
        map_tau_to_vars[curr_tau].append([key, value])
    
    # make list sorted by tau descending
    l_tau_tuple = []
    for curr_tau in map_tau_to_vars:
        l_tau_tuple.append((curr_tau, map_tau_to_vars[curr_tau]))
    l_tau_tuple.sort(key=lambda x: -float(x[0]))

    # for each tau setting, create tree for that tau's variables
    for _, tuples in l_tau_tuple:
        pruned_tree_data.append(create_tree(tree, check, tuples))
    return pruned_tree_data


def dump_pruned_tree(pruned_tree_data: List[Dict[str, Any]], output_dir: str):

    def collect_pruned_indices(
        node: ast_helper.Node
    ):
        if node.deleted:
            pruned_indices.append((node.start, node.end))
        else:
            for child in node.children:
                collect_pruned_indices(child)


    for pruned_tree_data in pruned_tree_data:
        global pruned_indices
        pruned_indices = []
        pruned_root = pruned_tree_data["entire_tree_with_deleted"]
        collect_pruned_indices(pruned_root.children[0])
        print("Pruned Indices: ", pruned_indices)
        # replace code[pruned_index[0]:pruned_index[1]] with <space>
        code = pruned_root.code
        pruned_code = ""
        cur = 0
        for pruned_index in pruned_indices:
            pruned_code += code[cur : pruned_index[0]] + " <PRUNED> "
            cur = pruned_index[1]
        
        if cur != len(code):
            pruned_code += code[cur :]
        pruned_code += '\n'

        with open(os.path.join(output_dir, "pruned_code.txt"), "a") as f:
            f.write(pruned_code)












