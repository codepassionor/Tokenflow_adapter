export XDG_CACHE_HOME=/root/autodl-tmp/cache_huggingface/; 
export NCCL_P2P_DISABLE="1";
export NCCL_IB_DISABLE="1";
export HF_ENDPOINT=https://hf-mirror.com;

# 运行 main.py 并传递第一个参数对
python main.py --begin=0 --end=300

# 运行 main.py 并传递第二个参数对
python main.py --begin=300 --end=600

# 运行 main.py 并传递第三个参数对
python main.py --begin=600 --end=900

# 运行 main.py 并传递第三个参数对
python main.py --begin=900 --end=1000