import ast
import traceback
from colorama import Fore, Back, Style
from typing import Optional


class Node:
    def __init__(self, code):
        self.code = code
        self.children = []
        self.start = 0
        self.end = 0
        self.intervals = []
        self.tokens = []
        self.logprobs = []
        self.nll = None
        self.deleted = None
        self.colon_name = None

    def __str__(self):
        def visit(n, pref):
            s = pref
            s += Fore.BLACK + Back.RED if n.deleted else Style.RESET_ALL
            s += f'"{n.code}"'
            s += str([self.code[t[0] : t[1]] for t in n.intervals])
            s += "[DELETED]" if n.deleted else ""
            s += Style.RESET_ALL
            s += "\n"
            for c in n.children:
                s += visit(c, pref + "\t")
            return s

        s = visit(self, "")
        return s

    def toJSON(self):
        return {
            "code": self.code,
            "start": self.start,
            "end": self.end,
            "intervals": self.intervals,
            "tokens": self.tokens,
            "logprobs": self.logprobs,
            "nll": self.nll,
            "deleted": self.deleted,
            "children": [c.toJSON() for c in self.children],
        }


def total_nodes(node: Node) -> int:
    sum_c = 0
    for c in node.children:
        sum_c += total_nodes(c)
    return 1 + sum_c


class CheckVisitor(ast.NodeVisitor):
    def __init__(self, code, print_flag=False):
        self.code = code
        self.check = True
        self.print_flag = print_flag
    
    def get_char_pos(self, code: str, node: ast.AST):
        lines = code.splitlines(keepends=True)
        start = sum(len(lines[i]) for i in range(node.lineno - 1)) + node.col_offset
        end = sum(len(lines[i]) for i in range(node.end_lineno - 1)) + node.end_col_offset
        return start, end

    def visit(self, node: ast.AST) -> Optional[Node]:
        if len(str(ast.unparse(node)).strip()) == 0:
            return None
        curr = Node(str(ast.unparse(node)))
        if hasattr(node, 'lineno') and hasattr(node, 'col_offset') and hasattr(node, 'end_lineno') and hasattr(node, 'end_col_offset'):
            curr.start, curr.end = self.get_char_pos(self.code, node)
        children_list = []
        children_code_list = []
        for n in ast.iter_child_nodes(node):
            c_node = self.visit(n)
            if c_node is not None:
                children_list.append(c_node)
                children_code_list.append(c_node.code)
        # sorted_children
        # if "int(d)" in children_code_list:
        #     print('here')
        curr.children = sorted(children_list, key=lambda child: child.start)
        return curr


def parse_with_pos(code):
    tree = ast.parse(code)
    
    class LineNumberAnnotator(ast.NodeVisitor):
        for node in ast.walk(tree):
            if not hasattr(node, 'lineno'):
                for child in ast.iter_child_nodes(node):
                    if hasattr(child, 'lineno'):
                        node.lineno = child.lineno
                        node.col_offset = child.col_offset
                        node.end_lineno = child.end_lineno
                        node.end_col_offset = child.end_col_offset
                        break
    
    LineNumberAnnotator().visit(tree)
    return tree




def get_node(code: str, print_traceback: bool = False) -> Optional[Node]:
    v = CheckVisitor(code)
    try:
        # t = ast.parse(code)
        t = parse_with_pos(code)
    except:
        if print_traceback:
            print(traceback.format_exc())
        return None
    return v.visit(t)


def get_num_nodes_from_code(code: str) -> int:
    v = CheckVisitor(code)
    try:
        t = ast.parse(code)
    except:
        return -1
    n = v.visit(t)
    return total_nodes(n)


def check_code(code: str) -> bool:
    v = CheckVisitor(code)
    t = ast.parse(code)
    n = v.visit(t)
    # print(n)
    return v.check

