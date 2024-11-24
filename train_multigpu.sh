export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5;                                                                      \
export NCCL_P2P_DISABLE=1;                           \
export NCCL_IB_DISABLE=1;                            \
export CUDA_LAUNCH_BLOCKING=1;                       \
export TORCH_USE_CUDA_DSA=1;                         \
accelerate launch                                                                       \
                    --num_machines 1                                                   \
                    --num_processes 1                                                  \
                    --gpu_ids 1                                                \
                    --num_cpu_threads_per_process 1                                                 \
    main_ddp.py
