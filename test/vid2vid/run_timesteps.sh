export XDG_CACHE_HOME=/root/autodl-tmp/cache_huggingface/; 
export NCCL_P2P_DISABLE="1";
export NCCL_IB_DISABLE="1";
export HF_ENDPOINT=https://hf-mirror.com;

# 运行 main.py 并传递第三个参数对
python main.py --begin=900 --end=1000