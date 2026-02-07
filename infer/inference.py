import os, sys

from utils.io_tools import Tools
from utils.dataloader import DataLoader
from utils.parser import Parser
from infer.llm import LLMFactory
from datetime import datetime
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer

import warnings
warnings.filterwarnings("ignore")

DATE_TIME_FORMAT = "%Y-%m-%d-%H-%M"
BATCH_SIZE = 10

LOCAL_MODEL_PATH: dict = {
    "deepseek-coder-33b-instruct": "/data1/model/deepseek/deepseek-ai/deepseek-coder-33b-instruct",
    "Qwen2.5-Coder-32B-Instruct": "/data1/model/qwen/Qwen/Qwen2.5-Coder-32B-Instruct",
    "Llama-3.1-70B-Instruct": "/data1/model/llama3/meta-llama/Llama-3.1-70B-Instruct",
}


def inference_on_tasks(args):
    data_path = f"../data/datasets/{args.dataset}.jsonl"
    
    try:
        # trust_remote_code = True trust_remote_code is true for Qwen, DeepSeek
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH[args.model_name_or_path], trust_remote_code=True)
        # LOCAL_MODEL_PATH[args.model_name_or_path]
    except Exception as e:
        print(f"Error: cannot load Tokenizer. Please check the model path. \nDetailed information: {e}")
    
    if not tokenizer.chat_template:
        print("Warning: The model does not have a 'chat_template' defined (possibly a base model).")
        print("Trying to use the default template, but this may result in suboptimal output.")


    prompts = DataLoader.load_prompt(args.dataset, args.prompt_type)
    factory = LLMFactory(args.model_name_or_path)
    llm = factory.get_llm(args)
    

    new_rule = '\"\"\" Please complete the above python code in following format:\n    ```python\n    ### your code here\n    ```\n    . Just give the completed function (including the function definition), do not generate extra codes like example usage or test cases.\n    \"\"\"\n\n'
    total_samples = []
    time_stamp = datetime.now().strftime(DATE_TIME_FORMAT)
    os.makedirs(args.output_dir, exist_ok=True)
    for i in tqdm(range(0, args.num_samples, BATCH_SIZE)):
        tasks = [
            (prompt["task_id"], prompt["prompt"] + new_rule)
            for _ in range(min(BATCH_SIZE, args.num_samples - i))
            for prompt in prompts
        ]
        # chat template
        for index in range(len(tasks)):
            messages = []
            user_query = tasks[index][1]
            messages.append({"role": "user", "content": user_query})
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,            
                    add_generation_prompt=True 
                )
                tasks[index] = (tasks[index][0], formatted_prompt)
            
            except Exception as e:
                print(f"Application template error: {e}\n(This usually means the model does not support chat_template, or the template needs to be specified manually)")
        

        samples = []
        error_sample = []
        if args.get_processed_data:
            samples, error_sample = llm.generate_with_parse(tasks)
        else:
            samples = llm.generate(tasks)
            samples = [
                Parser.extract_code_from_logprobs(prompts, sample) for sample in samples
            ]
        total_samples.extend(samples)
        if args.save_result:
            dump_path = os.path.join(
                args.output_dir,
                f"{args.dataset}-{args.prompt_type}-{args.model_name_or_path}-"
                f"{args.temperature}-{args.top_p}-n{args.num_samples}.jsonl",
            )
            error_dump_path = os.path.join(
                args.output_dir,
                f"error-{args.dataset}-{args.prompt_type}-{args.model_name_or_path}-"
                f"{args.temperature}-{args.top_p}-n{args.num_samples}.jsonl",
            )
            Tools.write_jsonl(dump_path, samples, append=True)
            Tools.write_jsonl(error_dump_path, error_sample, append=True)

            if args.get_processed_data:
                Tools.write_jsonl(
                    os.path.join(
                        args.output_dir,
                        f"processed-{args.dataset}-{args.prompt_type}-{args.model_name_or_path}-"
                        f"{args.temperature}-{args.top_p}-n{args.num_samples}.jsonl",
                    ),
                    DataLoader.load_solutions(
                        data_path,
                        dump_path,
                        args.num_samples,
                    ),
                    append=True,
                )
    return total_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HumanEval", type=str)
    parser.add_argument("--prompt_type", default="solution", type=str)
    parser.add_argument("--output_dir", default="./data/generated_data", type=str)
    parser.add_argument(
        "--model_name_or_path", default="deepseek-coder-33b-instruct", type=str
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--get_processed_data", action="store_true")
    parser.add_argument("--save_result", action="store_true")
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


if __name__ == "__main__":
    try:
        args = parse_args()
        inference_on_tasks(args)
    except OSError:
        pass
