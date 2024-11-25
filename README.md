##

### Training

#### 1. **Dataset Preparation (`make_dataset`)**
Before starting the training process, ensure the dataset is correctly prepared. Use the following script to organize and preprocess your data for training:

```bash
bash make_dataset.sh
```
This script handles dataset downloading, cleaning, and formatting to meet the training requirements.

---

#### 2. **Training ControlNet (`train_controlnet`)**
To train ControlNet using a single GPU, you can directly run the following script:

```bash
bash train_controlnet.sh
```
For multi-GPU training, make the necessary modifications in `train_controlnet.sh`, and refer to the example below:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --num_processes=4 --multi_gpu --num_machines=1 --gpu_ids=0,1,2,3 --num_cpu_threads_per_process 1 main_controlnet.py \
--pretrained_model_name_or_path /data/workspace/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9 \
--rank 4 --train_batch_size 1 --mixed_precision no
```

---

#### 3. **Training DDPM (`train_ddpm`)**
Use the script below for DDPM training:

```bash
bash train_ddpm.sh
```
To accelerate web connections, the `train_ddpm.sh` script uses `proxychains`. If you encounter connection issues, verify your proxy settings. For multi-GPU training, you can modify the script and use a similar setup as shown in the ControlNet example above.



# Project Announcement

We are excited to announce that following the acceptance of this paper, we will release the final checkpoint, testing code, bilateral filtering algorithm implementation, and a complete project webpage. 

Stay tuned for updates!

