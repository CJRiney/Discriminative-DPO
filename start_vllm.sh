#### Uncomment this to start vllm for phi-3-mini ####
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
        --model /mnt/data2/chris/research/bias-dpo/models/phi-3-mini-saved \
        --port 14231 \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --trust-remote-code

