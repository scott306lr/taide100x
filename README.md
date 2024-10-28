# TAIDE 100x
Accelerating TAIDE &amp; other Taiwan LLM models.

## To run the test script
```bash
LOGLEVEL=DEBUG CUDA_VISIBLE_DEVICES=0 python run_test.py --do-sample --temp 0.6 --max-new-tokens 256
```

<!-- ## To run the script to install flashinfer and rotary_emb from flash-attn
```bash
chmod +x install_flashinfer.sh'
./install_flashinfer.sh
``` -->

## To run the test script with flashinfer
```bash
LOGLEVEL=DEBUG CUDA_VISIBLE_DEVICES=0 python run_test.py --do-sample --temp 0.6 --max-new-tokens 256 --mode flashinfer
```