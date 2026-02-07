from typing import TypedDict, Dict, List, Tuple, Optional, Callable, Union, Any
import torch

# -------------------------------------Type Dict-------------------------------------#
Problem = TypedDict(
    "Problem",
    {
        "task_id": str,
        "description": str,
        "entry_point": str,
        "test": str,
        "canonical_solution": str,
    },
)

Solution = TypedDict(
    "Solution",
    {
        "task_id": str,
        "solution_id": int,
        "code": str,
        "mean_logprob": float,
        "ppl": float,
        "verbal_confidence": Optional[float],  # for verbalized confidence
    },
)

TestCase = TypedDict(
    "TestCase",
    {
        "task_id": str,
        "testcase_id": int,
        "raw": str,
        "input": str,
        "output": str,
    },
)


HypothesisThreshold = TypedDict(
    "HypothesisThreshold",
    {
        "name": str,
        "lambdas": torch.Tensor,
        "mono_increasing": bool,
    },
)

ExecResultPerTestcase = TypedDict(
    "ExecResultPerTestcase",
    {"testcase_id": int, "result": str, "output": Any},
)

ExecResult = TypedDict(
    "ExecResult",
    {
        "solution_id": int,
        "passed": bool,
        "ppl": float,
        "exec_result": List[ExecResultPerTestcase],
    },
)

Cluster = TypedDict(
    "Cluster",
    {
        "task_id": str,
        "exec_result": Tuple,
        "solutions_list": List[Solution],
        "total": int,
        "score": int,
        "passed": int,
    },
)


