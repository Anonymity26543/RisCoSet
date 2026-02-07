DATASET="HumanEval"
# DATASET="MBPP"
# DATASET="APPS"
DUMP_DIR="../output/raw_data"
MODEL_NAME_OR_PATH="deepseek-coder-33b-instruct"
# MODEL_NAME_OR_PATH="Qwen2.5-Coder-32B-Instruct"
# MODEL_NAME_OR_PATH="Llama-3.1-70B-Instruct"
TEMPERATURE=0.8
TOP_P=0.95
SAMPLE_NUM=256
SOURCE_FILE="inference.py"

CUDA_VISIBLE_DEVICES=0 \
python ${SOURCE_FILE} \
--dataset ${DATASET} \
--prompt_type solution \
--output_dir ${DUMP_DIR} \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--temperature ${TEMPERATURE} \
--top_p ${TOP_P} \
--num_samples ${SAMPLE_NUM} \
--max_tokens_per_call 4096 \
--seed 0 \
--save_result \
