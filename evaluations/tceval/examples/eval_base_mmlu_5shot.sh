set -e

### Breeze-7B-Base-v0_1
safetensor_model_path=MediaTek-Research/Breeze-7B-Base-v0_1
eval_batch=2
output_path=outputs/base/mmlu/5shot/Breeze-7B-Base-v0_1/
tensor_parallel_size=1

lm_eval --model vllm \
    --tasks mmlu \
    --model_args pretrained=${safetensor_model_path},trust_remote_code=True,tensor_parallel_size=${tensor_parallel_size},dtype=auto,gpu_memory_utilization=0.8 \
    --batch_size ${eval_batch} --output_path  ${output_path} \
    --num_fewshot 5 --log_samples --gen_kwargs temperature=0,top_k=0,top_p=0; 

python scripts/cal_lmeval.py mmlu ${output_path}
