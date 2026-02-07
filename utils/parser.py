import re
import ast
from utils.io_tools import Tools

STOP_SEQUENCES = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
# stop sequences from Codex
INPUTGEN_SIGN = "def input_generator()"
INPUTGEN_STOP_SEQUENCES = ["class", "def", "```", "if", "print", "assert"]


class Parser:
    @staticmethod
    def extract_code_from_logprobs(prompts: list[dict], solution: dict) -> str:
        """
        Extract the code from the given LOGPROBS.
        """
        tokens = solution["logprobs"]["tokens"]
        start_truncated_index = 0
        end_truncated_index = len(tokens)

        
        for i, token in enumerate(tokens):       
            if tokens[i] == "```" and tokens[i + 1] == "python" and tokens[i + 2] == "\n":
                start_truncated_index = i + 3 
                continue
            if tokens[i] == "``" and tokens[i + 1] == "`python" and tokens[i + 2] == "\n":
                start_truncated_index = i + 3 
                continue
            if tokens[i] == "```" or tokens[i] == "``":
                end_truncated_index = i
                break
        
        solution["logprobs"]["tokens"] = solution["logprobs"]["tokens"][
            start_truncated_index:end_truncated_index
        ]
        solution["logprobs"]["token_logprobs"] = solution["logprobs"][
            "token_logprobs"
        ][start_truncated_index:end_truncated_index]
        solution["completion"] = "".join(solution["logprobs"]["tokens"])
        solution["ppl"] = Tools.get_ppl_list(solution["logprobs"]["token_logprobs"])
        return solution

    @staticmethod
    def extract_normal_solution(solution: str) -> [str, int]:
        code = Parser._truncate(solution).split("assert")[0]
        length = len(code)
        code = Parser.repair_indent(code)
        return code, length

    @staticmethod
    def extract_solution(solution: dict, verbal: bool) -> [str, int]:
        """
        Extract correct code solution from given CODE.
        return CODE and relative LENGTH
        """
        if verbal:
            try:
                code_block = solution["completion"]
                if "```python" in solution["completion"]:
                    code_blocks = re.findall(
                        r"```python\n(.*?)```", code_block, re.DOTALL
                    )
                elif "```" in solution["completion"]:
                    code_blocks = re.findall(r"```(.*?)```", code_block, re.DOTALL)

                for block in code_blocks:
                    if f"def {solution['entry_point']}" in block:
                        code_block = block
                        break

                code_lines = code_block.strip().split("\n")
                start_index = [
                    i
                    for i in range(len(code_lines))
                    if code_lines[i].startswith(f"def {solution['entry_point']}")
                ][0]
                code = "\n".join(code_lines[start_index + 1 :])
            except BaseException:
                code = ""
        else:
            code = solution["completion"]

        code = Parser._truncate(code).split("assert")[0]
        length = len(code)
        code = Parser.repair_indent(code)
        # code = Parser.repair_indent(Parser._truncate(code))
        # code = code.split('assert')[0]
        return code, length

    def extract_verbal_confidence(completion: str) -> float:
        try:
            conf = re.search(r"Confidence: (\d+)%", completion).group(1)
        except AttributeError:
            conf = 50.0
        return float(conf) / 100.0

    @staticmethod
    def extract_testcase(testcase: str, entry_point: str) -> list[str]:
        """
        Extract correct testcase from given TESTCASE according to given ENTRY_POINT.
        """
        split_by_assert = [
            f"assert {part}".strip()
            for part in f"assert {testcase}".split("assert ")
            if (entry_point.strip() in part) and len(part.strip()) > 0
        ]
        truncated_test_cases = [Parser._truncate(i) for i in split_by_assert]
        checked_assertions = [
            i for i in truncated_test_cases if Parser._check_test_case_validation(i)
        ]
        return checked_assertions

    @staticmethod
    def extract_testcase_with_io(completion: str, entry_point: str) -> list[str]:
        """
        Extract correct testcase from given TESTCASE according to given ENTRY_POINT.
        """
        visitor = AssertVisitor()
        return visitor(
            Parser._clean_truncated_code("assert " + completion), entry_point
        )

    @staticmethod
    def extract_code(content: str):
        code = ""
        if "```" in content:
            if "python" in content.lower():
                # code = re.search(r"```python(?P<code>[^`]*)(```|$)", content, flags=re.IGNORECASE).group('code')
                code = re.search(
                    r"```python(?P<code>[^`]*)(```|$)", content, flags=re.IGNORECASE
                )
                code = code.group("code") if code else ""
            else:
                code = re.search(
                    r"```(?P<code>[^`]*)(```|$)", content, flags=re.IGNORECASE
                ).group("code")
        else:
            code = content
        code = re.sub(r"import.*\n", "", code)
        code = re.sub(r"from.*\n", "", code)
        return code

    @staticmethod
    def extract_code_from_inputgen(code: str) -> str:
        """
        Extract the code from the given CODE.
        """
        if "# END SOLUTION" in code:
            code = code.split("# END SOLUTION")[0]
        if INPUTGEN_SIGN in code:
            code = code.split(INPUTGEN_SIGN)[1]
            for stop_sequence in INPUTGEN_STOP_SEQUENCES:
                if stop_sequence in code:
                    code = code.split(stop_sequence)[0]
            code = INPUTGEN_SIGN + code
            return Parser.repair_indent(code.strip())
        return ""

    @staticmethod
    def _truncate(content: str) -> str:
        """
        Truncates a string before specific STOP_SEQUENCES.
        """
        for stop_sequence in STOP_SEQUENCES:
            if stop_sequence in content:
                content = content.split(stop_sequence)[0]
        return content.strip()

    @staticmethod
    def _clean_truncated_code(code: str) -> str:
        """
        Clean up the truncated code and keep the syntactically correct parts.
        Almost all trauncation happened at the tail, so the mean time complexity is O(1).
        """
        lines = code.split("\n")

        for i in range(len(lines), 0, -1):
            temp_code = "\n".join(lines[:i])
            try:
                ast.parse(temp_code)
                return temp_code
            except SyntaxError:
                continue

        return ""

    @staticmethod
    def repair_indent(code: str) -> str:
        """
        Repair the indent of the given CODE.
        """
        blank = "    "
        if code.startswith("def"):
            return blank + code
        if not code.startswith(blank):
            code = blank + code
            if len(code.split("\n")) > 1 and not code.split("\n")[1].startswith(blank):
                code = code.replace("\n", "\n" + blank)
        return code

    @staticmethod
    def _check_test_case_validation(test_case):
        """
        Checks if the given TESTCASE is valid by compile it.
        """
        if len(test_case.strip()) < 1:
            return False
        if "assert" not in test_case:
            return False
        try:
            multi_line_test_case = test_case.replace("\n", "\n    ")
            assert_in_a_block = f"try:\n    {multi_line_test_case}\nexcept:\n    pass\n"
            compile(assert_in_a_block, "", "exec")
            return True
        except Exception:
            return False


class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.func = None
        self.entry_point = ""

    def visit_Call(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        if isinstance(node.func, ast.Name) and node.func.id == self.entry_point:
            self.func = node

    def __call__(self, entry_point: str, code: str) -> str:
        self.func = None
        self.entry_point = entry_point
        self.visit(ast.parse(code))
        return ast.unparse(self.func) if self.func is not None else None


class AssertVisitor(ast.NodeVisitor):
    def __init__(self):
        self.entry_point = ""
        self.raw = []
        self.input = []
        self.output = []

    def visit_Assert(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        raw_assert = ast.unparse(node)
        if (
            isinstance(node.test, ast.Compare)
            and self.entry_point in raw_assert
            and len(node.test.ops) > 0
            and isinstance(node.test.ops[0], ast.Eq)
        ):
            self.raw.append(raw_assert)
            left = ast.unparse(node.test.left)
            right = ast.unparse(node.test.comparators[0])

            if self.entry_point in left:
                self.input.append(left)
                self.output.append(right)
            elif self.entry_point in right:
                self.input.append(right)
                self.output.append(left)

    def __call__(self, code: str, entry_point) -> list:
        self.entry_point = entry_point
        try:
            self.visit(ast.parse(code))
        except SyntaxError as e:
            print(f"SyntaxError: {e}")
            print(f"code: \n{code}")
            print("------------------------------------------")
            return []
        return [
            dict(raw=r, input=i, output=o)
            for r, i, o in zip(self.raw, self.input, self.output)
        ]
