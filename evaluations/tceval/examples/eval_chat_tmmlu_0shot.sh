set -e 

### Breeze-7B-Instruct-v0_1
safetensor_model_path=MediaTek-Research/Breeze-7B-Instruct-v0_1
template=breeze
sys="You are a helpful AI assistant."
eval_batch=2
output_path=outputs/chat/tmmlu/0shot/Breeze-7B-Instruct-v0_1/
tensor_parallel_size=1

lm_eval --model vllm \
    --tasks tmmluplus_fewshot \
    --conv_template ${template} --system_message "${sys}" --fewshot_method chat_iteration \
    --model_args pretrained=${safetensor_model_path},trust_remote_code=True,tensor_parallel_size=${tensor_parallel_size},dtype=auto,gpu_memory_utilization=0.8 \
    --batch_size ${eval_batch} --output_path  ${output_path} \
    --num_fewshot 0 --log_samples --gen_kwargs temperature=0,top_k=0,top_p=0; 

python scripts/cal_lmeval.py tmmlu ${output_path}
