export CUDA_VISIBLE_DEVICES=0;                                                                      \
proxychains accelerate launch                                                                       \
                    --num_processes=1                                                               \
                    --num_machines=1                                                                \
                    --gpu_ids=0                                                                     \
                    --num_cpu_threads_per_process 1                                                 \
    main.py --pretrained_model_name_or_path     runwayml/stable-diffusion-v1-5                      \
            --rank                              4                                                   \
            --train_batch_size                  1                                                   \
            --mixed_precision                   no                                                  \
            --dataset_src                       /root/autodl-tmp/data/msrvtt-webdataset.tar         \
            --cache_dir                         /root/autodl-tmp/cache                              \
            --train_dataset_size                10000