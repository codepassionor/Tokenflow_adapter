##

# Enhancing Low-Cost Video Editing with Lightweight Adaptors and Temporal-Aware Inversion

## Authors
Yangfan He, Sida Li, Kun Li, Jianhui Wang, Binxu Li, Tianyu Shi, Jun Yin, Miao Zhang
## Getting Started
1. Clone this repository:
```bash
git clone https://github.com/codepassionor/Tokenflow_adapter.git
```
2. Install the dependencies:
```bash
conda create --name myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

## Training

#### 1. **Dataset Preparation (`make_dataset`)**
Before starting the training process, ensure the dataset is correctly prepared. Please download the MSRVTT dataset locally first, you can get it through this link [MRSVTT](https://drive.google.com/file/d/15bBcfrCxr27XpTABX8Oy7fn09QYs45Cq/view?usp=drive_link).

Then use the following script to organize and preprocess your data for training:
```bash
python make_dataset/msrvtt-depth-map.py
```
This script handles dataset cleaning, and formatting to meet the training requirements.

---

#### 2. **Training ControlNet (`train_controlnet`)**
To train ControlNet using a single GPU, you can directly run the following script:

```bash
bash train_controlnet.sh
```
For multi-GPU training, make the necessary modifications in `train_controlnet.sh`, and refer to the example below:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch     \
                    --num_machines 1  \
                    --num_processes 1 \
                    --gpu_ids 1       \
                    --num_cpu_threads_per_process 1 \
    main_controlnet.py 
--pretrained_model_name_or_path /data/workspace/huggingface/hub/runwayml/stable-diffusion-v1-5        --rank 4  --train_batch_size 1  --mixed_precision no
```

---

#### 3. **Training DDPM (`main_ddp`)**
Use the script below for DDPM training:

```bash
bash train_multigpu.sh
```
To accelerate web connections, the `train_ddpm.sh` script uses `proxychains`. If you encounter connection issues, verify your proxy settings. For multi-GPU training, you can modify the script and use a similar setup as shown in the ControlNet example above.


## Visualization
This section shows some visualization results on downstream algorithms.

<img src="./assert/output_1.gif" width = 450>
<img src="./assert/output_2.gif" width = 600>

We provide a comparison of the effects of different timesteps of the training and inference process for you to pick the desired parameters.

<img src="./assert/timestep.png" width = 600>


## Todo List

- [x] Testing Code  
- [ ] Open-Sourcing Much More Visualization Results
- [ ] Project webpage
- [ ] More detailed running instructions

## Citation
If you found this repository useful, please consider citing our paper:
@article{he2025enhancing,
  title={Enhancing Low-Cost Video Editing with Lightweight Adaptors and Temporal-Aware Inversion},
  author={He, Yangfan and Li, Sida and Li, Kun and Wang, Jianhui and Li, Binxu and Shi, Tianyu and Yin, Jun and Zhang, Miao and Wang, Xueqian},
  journal={arXiv preprint arXiv:2501.04606},
  year={2025}
}
