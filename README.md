# TAIDE 100x
Accelerating TAIDE &amp; other Taiwan LLM models.

## To run the test script
```bash
LOGLEVEL=DEBUG CUDA_VISIBLE_DEVICES=0 python run_test.py --do-sample --temp 0.6 --max-new-tokens 256
```

## To run the test script with flashinfer
```bash
LOGLEVEL=DEBUG CUDA_VISIBLE_DEVICES=0 python run_test.py --do-sample --temp 0.6 --max-new-tokens 256 --mode flashinfer
```