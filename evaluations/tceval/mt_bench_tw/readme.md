# MT Bench TW

We made minor changes to the original [MT Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) to build MT-Bench-TW


## Installation
```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"
```

## Evaluate a model on MT-bench-TW

### Step 1. Generate model answers to MT-bench-TW questions
```
python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]
```
Arguments:
  - `[MODEL-PATH]` is the path to the weights, which can be a local folder or a Hugging Face repo ID.
  - `[MODEL-ID]` is a name you give to the model.

e.g.,
```
python gen_model_answer.py --model-path MediaTek-Research/Breeze-7B-Instruct-v1_0 --model-id Breeze-7B
```
The answers will be saved to `data/mt_bench_tw/model_answer/[MODEL-ID].jsonl`.

The script leverages `vllm` for efficient inference.

To make sure the prompt template is incorporated correctly, please see `conversation.py` (ported from FastChat).

You can also specify `--num-gpus-per-model` for model parallelism (needed for large 65B models) and `--num-gpus-total` to parallelize answer generation with multiple GPUs.

### Step 2. Generate GPT-4 judgments
There are several options to use GPT-4 as a judge, such as pairwise winrate and single-answer grading.
In MT-bench, we recommend single-answer grading as the default mode.
This mode asks GPT-4 to grade and give a score to model's answer directly without pairwise comparison.
For each turn, GPT-4 will give a score on a scale of 10. We then compute the average score on all turns.

```
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
```

e.g.,
```
python gen_judgment.py --model-list Breeze-7B gpt3.5 --parallel 2
```
The judgments will be saved to `data/mt_bench_tw/model_judgment/gpt-4_single.jsonl`


### Step 3. Visuzlize results
```
python3 qa_browser.py
```