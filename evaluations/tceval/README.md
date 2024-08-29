# TCEval v2

## Install

```bash
cd lm-evaluation-harness_mr-revised
pip3 install -e ".[vllm]"
pip3 install -U vllm
cd ..
```

## Evaluate Local Models (MMLU, TMMLU+, and Penguin_Table)

please reference [examples](./examples/)

## Evaluate API Models (MMLU, TMMLU+, and Penguin_Table)

please check [scripts/cal_likelihood_by_api.py](scripts/cal_likelihood_by_api.py)

## Evaluate MTBench-tw

please reference [here](./mt_bench_tw/).
