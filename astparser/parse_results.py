import os
import sys
from typing import List
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)

from astparser import ast_helper



def populate_start_and_end(parent: ast_helper.Node) -> None:
    curr_ind = 0
    last_child = ""
    # child_se_record = []
    # assumes parent.children are in order of presence
    for child in parent.children:
        try:
            # new rule
            if last_child == "Tuple" and child.code[0] == '(' and child.code[-1] == ')':
                child.code = '[' + child.code[1:-1] + ']'
                
            index_in_truncated_parent = parent.code[curr_ind:].index(child.code)
            child.start = parent.start + curr_ind + index_in_truncated_parent
            child.end = child.start + len(child.code)
            # child_se_record.append(tuple((child.start, child.end)))
            curr_ind += index_in_truncated_parent + len(child.code)
            last_child = child.code
   
        except:
            # print('parent.code: ', parent.code)
            # print('last_child: ', last_child)
            # print('child.code: ', child.code)
            print('Error: not find string')
                
    for child in parent.children:
        populate_start_and_end(child)


def assert_start_end_are_correct(parent: ast_helper.Node, addtl: str) -> None:
    assert addtl[parent.start : parent.end] == parent.code
    for c in parent.children:
        assert_start_end_are_correct(c, addtl)


def make_dependent(parent: ast_helper.Node) -> None:
    # window start
    prev_end = parent.start
    for i in range(len(parent.children)):
        if parent.children[i].start - prev_end > 0:
            parent.intervals.append((prev_end, parent.children[i].start))
        prev_end = parent.children[i].end
    # last window end
    # if len(parent.intervals) > 0 and parent.children[-1].end < parent.end:
    if len(parent.children) > 0 and parent.children[-1].end < parent.end:
        parent.intervals.append((parent.children[-1].end, parent.end))
    # case where no children
    if len(parent.children) == 0:
        parent.intervals.append((parent.start, parent.end))
    for c in parent.children:
        make_dependent(c)


def remove_all_spaces_in_code(parent: ast_helper.Node) -> ast_helper.Node:
    parent.code = parent.code.replace(" ", "")
    for i in range(len(parent.children)):
        parent.children[i] = remove_all_spaces_in_code(parent.children[i])
    return parent


def code_to_final_ast(code: str) -> ast_helper.Node:
    root = ast_helper.get_node(code, True)
    try:
        remove_all_spaces_in_code(root)
        populate_start_and_end(root)
        root.end = len(root.code)
        assert_start_end_are_correct(root, root.code)
        make_dependent(root)
    except Exception as e:
        print(f"Syntax Error: {e}")
        return None
    return root


def intervals_to_token_probs(
    parent: ast_helper.Node,
    map_index_to_token_ind: dict,
    tokens: List[str],
    token_logprobs: List[float],
) -> None:
    token_inds = []
    for tup in parent.intervals:
        for output_index in range(tup[0], tup[1]):
            if output_index in map_index_to_token_ind.keys():
                token_inds.append(map_index_to_token_ind[output_index])
    token_inds = sorted(list(set(token_inds)))
    for token_ind in token_inds:
        parent.tokens.append(tokens[token_ind])
        parent.logprobs.append(token_logprobs[token_ind])
    
    # APPS
    # parent.nll = max(-1 * sum(parent.logprobs), 1e-2) / 10.0
    
    # Others
    parent.nll = max(-1 * sum(parent.logprobs), 1e-2)
        
    for c in parent.children:
        intervals_to_token_probs(c, map_index_to_token_ind, tokens, token_logprobs)


def min_edit_operations(str1, str2):
    n, m = len(str1), len(str2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    ops_rev = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and str1[i - 1] == str2[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops_rev.append(('delete', i - 1, None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops_rev.append(('insert', i, str2[j - 1])) 
            j -= 1
        else:
            ops_rev.append(('replace', i - 1, str2[j - 1]))
            i -= 1
            j -= 1

    ops = ops_rev[::-1]

    s = list(str1)                     
    pos = list(range(n))            
    insert_count = {}                 
    output = []                     

    for op in ops:
        if op[0] == 'replace':
            p, c = op[1], op[2]
            idx = pos[p]
            output.append(('replace', idx))
            s[idx] = c
        elif op[0] == 'delete':
            p = op[1]
            idx = pos[p]
            output.append(('delete', idx))
            del s[idx]
            for k in range(p + 1, n):
                pos[k] -= 1
        else:  
            p, c = op[1], op[2]
            if p == n:
                idx = len(s)
            else:
                if p == 0:
                    base = 0
                else:
                    base = pos[p - 1] + 1
                k = insert_count.get(p, 0)
                idx = base + k
                insert_count[p] = k + 1
            output.append(('insert', idx))
            s.insert(idx, c)
            for k in range(p, n):
                pos[k] += 1

    return output


def check_string(str1, str2, index_map):
    for i in range(len(index_map)):
        if index_map[i] >= 0:
            if str1[i] != str2[index_map[i]]:
                return False
        
    return True


def add_probability_to_nodes(
    root: ast_helper.Node, response: dict, debug: bool = False
) -> None:
    # remove all spaces in tokens
    response["logprobs"]["tokens"] = [
        tok.replace(" ", "") for tok in response["logprobs"]["tokens"]
    ]
    response["logprobs"]["tokens"] = [
        tok.replace("\t", "") for tok in response["logprobs"]["tokens"]
    ]
    # response["logprobs"]["tokens"] = [
    #     tok.replace("\n", "") for tok in response["logprobs"]["tokens"]
    # ]
    
    response_code = "".join(response["logprobs"]["tokens"])
    tree_code = root.code
    
    ast_index_map = np.array(list(range(len(response_code))))
    if response_code != tree_code:
        
        ops = min_edit_operations(response_code, tree_code)
        # print(ops)
        for op in ops:
            if op[0] == 'replace':
                raise ValueError("use replacement ops!")
            elif op[0] == 'delete':
                if op[1] > np.max(ast_index_map):
                    continue
                idx = int(np.where(ast_index_map == op[1])[0])
                ast_index_map[idx:] = ast_index_map[idx:] - 1
                ast_index_map[idx] = -10000
            elif op[0] == 'insert':
                if op[1] > np.max(ast_index_map):
                    continue
                idx = int(np.where(ast_index_map == op[1])[0])
                ast_index_map[idx:] = ast_index_map[idx:] + 1

    # check string is consisent
    assert check_string(response_code, tree_code, ast_index_map)
    
    map_index_to_token_ind = {}
    output_index = 0
    for i in range(len(response["logprobs"]["tokens"])):
        for _ in range(len(response["logprobs"]["tokens"][i])):
            if ast_index_map[output_index] != -10000:
                map_index_to_token_ind[ast_index_map[output_index]] = i
            output_index += 1
            
    if debug:
        for key in map_index_to_token_ind:
            print(
                key, "->", response["logprobs"]["tokens"][map_index_to_token_ind[key]]
            )
        print("----")
        for i in range(len(root.code)):
            print(i, "->", root.code[i])
        print("----")
    
    intervals_to_token_probs(
        root,
        map_index_to_token_ind,
        response["logprobs"]["tokens"],
        response["logprobs"]["token_logprobs"],
    )


