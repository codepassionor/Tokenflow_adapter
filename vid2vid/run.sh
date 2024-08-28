export XDG_CACHE_HOME=/root/autodl-tmp/cache_huggingface/; 
export NCCL_P2P_DISABLE="1";
export NCCL_IB_DISABLE="1";
export HF_ENDPOINT=https://hf-mirror.com; python main.py