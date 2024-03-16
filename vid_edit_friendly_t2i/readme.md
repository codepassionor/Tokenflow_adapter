##

### training
```
export CUDA_VISIBLE_DEVICES=0,1,2,3; accelerate launch --num_processes=4 --multi_gpu --num_machines=1 --gpu_ids=0,1,2,3 --num_cpu_threads_per_process 1  main.py --pretrained_model_name_or_path /data/workspace/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9  --rank 4 --train_batch_size 1 --mixed_precision no
```

### For single gpu locally
```
export CUDA_VISIBLE_DEVICES=0; accelerate launch --num_processes=1 --num_machines=1 --gpu_ids=0 --num_cpu_threads_per_process 1  main.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5  --rank 4 --train_batch_size 2 --mixed_precision no --dataset_src /home/shuyumo/research/t2v/data/MSRVTT
```

TODO
- realdataset: redesign dataset / dataloader structure, which current version is not appropriate.
- loss implement
- FP16 support
- logging


### core structure
1. hook_test.py for saving inmediate feature in unet
2. vid_edit_dataset implement dataset and dataloader, current implement a toy random dataset only
3. main.py, training, add our loss in line 699





