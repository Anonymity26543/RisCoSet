import os
from typing import List, Dict
from vllm import LLM, SamplingParams

from utils.io_tools import Tools
from utils.parser import Parser


class LLMGen:
    """
    LLM generation base model, which can be inherited to LLMLocal model.
    """

    def __init__(self) -> None:
        pass

    def generate(self, tasks: List[Dict]) -> List[Dict]:
        """
        Given a list of generation tasks TASKS, which contains (task_id, prompt),
        return a list of generated samples SAMPLES.
        """
        raise NotImplemented


class LLMLocalGen(LLMGen):
    def __init__(self, model_name: str, args: Dict) -> None:
        super().__init__()
        available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        # available_gpus = ["cuda:0"]
        self.llm = LLM(
            model_name,
            tensor_parallel_size=len(available_gpus),
            max_model_len=8192,
            gpu_memory_utilization=0.8,
        )
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            logprobs=1,
        )

    def generate(self, tasks: List[Dict]) -> List[Dict]:
        # with torch.no_grad():
        outputs = self.llm.generate([item[1] for item in tasks], self.sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        samples = [
            {
                "task_id": task_id,
                "completion": output.outputs[0].text,
                "logprobs": {
                    "tokens": [
                        list(x.values())[0].decoded_token
                        for x in output.outputs[0].logprobs
                    ],
                    "token_logprobs": [
                        list(x.values())[0].logprob for x in output.outputs[0].logprobs
                    ],
                },
                # "ppl": Tools.get_ppl(output.outputs[0].logprobs),
            }
            for task_id, output in zip([item[0] for item in tasks], outputs)
        ]
        return samples

    def generate_with_parse(self, tasks: List[Dict]) -> [List[Dict], List[Dict]]:
        outputs = self.llm.generate([item[1] for item in tasks], self.sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))

        samples = []
        error_sample = []
        for task_id, output in zip([item[0] for item in tasks], outputs):
            processed_code, code_len = Parser.extract_normal_solution(
                output.outputs[0].text.replace("\t", "    ")
            )
            if code_len == 0:
                error_sample.append(
                    {
                        "task_id": task_id,
                        "completion": output.outputs[0].text.replace("\t", "    "),
                        #'logprobs': output.outputs[0].logprobs
                    }
                )
            else:
                samples.append(
                    {
                        "task_id": task_id,
                        "completion": processed_code,
                        "logprobs_len": Tools.get_len_logprobs(
                            output.outputs[0].logprobs[:code_len]
                        ),
                        "mean_logprob": Tools.get_mean_logprobs(
                            output.outputs[0].logprobs[:code_len]
                        ),
                        "ppl": Tools.get_ppl(output.outputs[0].logprobs[:code_len]),
                    }
                )
        return samples, error_sample


class LLMFactory:
    """
    A LLM instantiation obeject used to instantiate different kinds of LLM instances.
    """

    LOCAL_MODEL_PATH: dict = {
        "deepseek-coder-33b-instruct": "/data1/model/deepseek/deepseek-ai/deepseek-coder-33b-instruct",
        "Qwen2.5-Coder-32B-Instruct": "/data1/model/qwen/Qwen/Qwen2.5-Coder-32B-Instruct",
        "Llama-3.1-70B-Instruct": "/data1/model/llama3/meta-llama/Llama-3.1-70B-Instruct",
    }
    

    def __init__(self, model_name_or_path: str) -> None:
        """
        Return a LLM factory object for instantiating LLM object.
        """
        if model_name_or_path in self.LOCAL_MODEL_PATH:
            self.llm_type = LLMLocalGen
            self.model_name = self.LOCAL_MODEL_PATH[model_name_or_path]
        else:
            raise ValueError(
                f"Invalid LLM type! Supported LLMs {self.LOCAL_MODEL_PATH.keys()}"
            )

    def get_llm(self, args) -> LLMGen:
        return self.llm_type(self.model_name, args)
