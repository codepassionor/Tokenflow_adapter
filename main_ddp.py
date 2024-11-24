#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""
from itertools import cycle, chain
import io
import torchvision
from torchvision import transforms
import webdataset as wds
from utils.vid_edit_dataset import get_data_loader
from utils.prompt_learner import PrefixToken
from utils.hook_test import FeatureExtractor
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import utils.preprocess
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
import torch.nn.init as init
import torch.nn as nn

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import itertools
import pickle


#--use_8bit_adam  \

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

torch.multiprocessing.set_start_method("spawn", force=True)


logger = get_logger(__name__, log_level="INFO")

def save_model_card(
    repo_id: str,
    images: list = None,
    base_model: str = None,
    dataset_name: str = None,
    repo_folder: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/root/autodl-tmp/cache_huggingface/huggingface/hub/models--runwayml--stable-diffusion-v1-5/",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/root/autodl-tmp/cache",
        help="Path to cache pretrained model (must with large amount of free space)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    #parser.add_argument(
    #    "--dataset_name",
    #    type=str,
    #    default=None,
    #    help=(
    #        "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
    #        " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
    #        " or to a folder containing files that 🤗 Datasets can understand."
    #    ),
    #)
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    #parser.add_argument(
    #    "--train_data_dir",
    #    type=str,
    #    default=None,
    #    help=(
    #        "A folder containing the training data. Folder contents must follow the structure described in"
    #        " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
    #        " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
    #    ),
    #)
    #parser.add_argument(
    #    "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    #)
    #parser.add_argument(
    #    "--caption_column",
    #    type=str,
    #    default="text",
    #    help="The column of the dataset containing a caption or a list of captions.",
    #)
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--prompt_threshold",
        type=float,
        default=1.0,
        help=(
            "Used to control the proportion of pre-trained prompts used instead of direct coding"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=8)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.01,
        help=("our loss ratio"),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    return args

@torch.no_grad()
def encode_prompt(prompt_batch, text_encoder, tokenizer):
    with torch.no_grad():
        text_inputs = tokenizer(
            prompt_batch,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )
    # print('qwq1', prompt_embeds[0].shape)
    # print('qwq2', prompt_embeds[1].shape)
    return prompt_embeds[0]

@torch.no_grad()
def ddim_inversion(unet, scheduler, cond, latent_frames, batch_size, timesteps_to_save=None):
    timesteps = reversed(scheduler.timesteps)
    for i, t in enumerate(tqdm(timesteps)):
        for b in range(0, latent_frames.shape[0], batch_size):
            x_batch = latent_frames[b:b + batch_size]
            model_input = x_batch
            cond_batch = cond[b:b + batch_size]

            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else scheduler.final_alpha_cumprod
            )

            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = unet(model_input, t, encoder_hidden_states=cond_batch).sample
            pred_x0 = (x_batch - sigma_prev * eps) / mu_prev
            latent_frames[b:b + batch_size] = mu * pred_x0 + sigma * eps

    return latent_frames, eps 

@torch.no_grad()
def get_text_embeds(tokenizer, text_encoder, prompt, negative_prompt, device="cuda"):
        text_input = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = tokenizer(negative_prompt, padding='max_length', max_length=tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

def bilateral_filter_1d_vectorized(input, kernel_size, sigma_spatial, sigma_intensity):
    """
    Apply a vectorized bilateral filter on 1D input data.

    Args:
    - input (torch.Tensor): The input tensor of shape (batch_size, channels, width).
    - kernel_size (int): The size of the filter kernel.
    - sigma_spatial (float): The spatial distance standard deviation.
    - sigma_intensity (float): The intensity difference standard deviation.

    Returns:
    - filtered_output (torch.Tensor): The output after applying the bilateral filter.
    """
    half_kernel = kernel_size // 2
    batch_size, channels, width = input.shape

    # Pad the input to handle the borders
    padded_input = F.pad(input, (half_kernel, half_kernel), mode='reflect')

    # Create the spatial Gaussian kernel (same for all data points)
    spatial_gaussian = torch.exp(-torch.pow(torch.arange(-half_kernel, half_kernel + 1, device=input.device, dtype=input.dtype), 2) / (2 * sigma_spatial ** 2))

    # Create a sliding window of the local regions for each element in the input
    local_regions = padded_input.unfold(2, kernel_size, 1)  # (batch_size, channels, width, kernel_size)

    # Compute the intensity differences (vectorized for all elements)
    center_pixel = input.unsqueeze(-1)  # Shape: (batch_size, channels, width, 1)
    intensity_gaussian = torch.exp(-torch.pow(local_regions - center_pixel, 2) / (2 * sigma_intensity ** 2))

    # Combine spatial and intensity components
    bilateral_kernel = spatial_gaussian.view(1, 1, 1, -1) * intensity_gaussian

    # Normalize the kernel
    bilateral_kernel = bilateral_kernel / bilateral_kernel.sum(dim=-1, keepdim=True)

    # Apply the kernel to the local regions
    filtered_output = (bilateral_kernel * local_regions).sum(dim=-1)

    return filtered_output

class FeatureExtractor_inv:
    def __init__(self, model):
        self.model = model
        self.features = []
        self.hooks = []
        self.register_hook()

    def hook_function(self, module, input, output):
        #return torch.flip(output, [0]) # this is a extreme case, but still work in low CFG case
        # B, N, d
        output = output.permute(1, 2, 0)
        # N，d, B
        output = bilateral_filter_1d_vectorized(output, 5, 1, 1)
        output = output.permute(2, 0, 1)

        return output

    def register_hook(self):
        #print(self.model._modules)
        #res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
        res_dict = {3: [0, 1]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
        unet = self.model
        for res in res_dict:
            for block in res_dict[res]:
                module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
                self.hooks.append(module.register_forward_hook(self.hook_function))

    def get_features(self, input_data):
        _ = self.model(input_data)
        return self.features

    def reset(self):
        self.features = []

    def remove_hook(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def main():
    # Video2webdatset()
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    
    controlnet_path = "/root/autodl-tmp/lora_fs/vid_multigpu/controlnet-model/checkpoint-100000"
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer",
        revision=args.revision, cache_dir=args.cache_dir
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, cache_dir=args.cache_dir
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant, cache_dir=args.cache_dir
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
        revision=args.revision, variant=args.variant, cache_dir=args.cache_dir
    )
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, subfolder="controlnet",
        revision=args.revision, variant=args.variant, cache_dir=args.cache_dir
    )
    clip_model = CLIPModel.from_pretrained("/root/autodl-tmp/cache_huggingface/huggingface/hub/openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("/root/autodl-tmp/cache_huggingface/huggingface/hub/openai/clip-vit-base-patch32")
    pretrained_path = "/root/autodl-tmp/lora_fs/vid_multigpu/prompt_learner/prefix_token_model_1.pth"
    loaded_prompt_learner = PrefixToken(4)            #.load(pretrained_path, accelerator.device)
    #loaded_prompt_learner.load(pretrained_path, unet.device)
    # freeze parameters of models to save more memory
    vae.requires_grad_(False)
    clip_model.requires_grad_(False)
    # Freeze the unet parameters before adding adapters
    for param_u in unet.parameters():
        param_u.requires_grad_(False)
    for param_c in controlnet.parameters():
        param_c.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    class CustomLoraConfig(LoraConfig):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def apply_to(self, model):
            for name, module in model.named_modules():
                if "down_blocks" in name and any(target in name for target in self.target_modules):
                    self.add_adapter(module)

    # 配置并添加 LoRA adapter
    unet_lora_config = CustomLoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)  #disable_adapters()  unet.enable_adapters()
    unet.enable_adapters()

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    clip_model.to(accelerator.device, dtype=weight_dtype)
    loaded_prompt_learner.to(accelerator.device, dtype=weight_dtype)

    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)
        cast_training_params(controlnet, dtype=torch.float32)
        cast_training_params(loaded_prompt_learner, dtype=torch.float32)
        cast_training_params(clip_model, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    #lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        #controlnet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    #unet_params = filter(lambda p: p.requires_grad, unet.parameters())
    #controlnet_params = filter(lambda p: p.requires_grad, controlnet.parameters())
    adapter_para = filter(lambda p: p.requires_grad, unet.parameters()) #chain(unet_params, controlnet_params)
    prefix_token_para = filter(lambda p: p.requires_grad, loaded_prompt_learner.parameters())
    all_params = itertools.chain(adapter_para, prefix_token_para)

    optimizer = optimizer_cls(
        all_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # TODO: [SYM]:
    # fingerprint used by the cache for the other processes to load the result
    # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401

    # DataLoaders creation:
    # [SYM]: Can not use batch in webdataset, which is inconsistant with accelerator.
    # @See: https://github.com/huggingface/accelerate/issues/1370
    # @See: https://github.com/huggingface/accelerate/issues?q=is%3Aissue+is%3Aopen+webdataset

    train_dataloader = get_data_loader(args)
    dataloader_length = 10000 // args.train_batch_size

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(dataloader_length / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    #hook = hook_test.DistFeatureExtractorV2(unet, accelerator)
    hook = FeatureExtractor(unet)
    #hooker = FeatureExtractor_inv(unet)

    # Prepare everything with our `accelerator`.
    unet, controlnet, optimizer, train_dataloader, lr_scheduler, loaded_prompt_learner, clip_model, clip_processor = accelerator.prepare(
        unet, controlnet, optimizer, train_dataloader, lr_scheduler, loaded_prompt_learner, clip_model, clip_processor
    )

    # HOOK

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(dataloader_length / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))
    
    '''
    # 修改1：Train之前用ddim_inversion为每个视频帧保存噪声
    #print("开始修改1")
    noise_list = []  # 用于存储每个视频帧的噪声
    '''

    '''
    # 生成 video_frames ，包含所有视频帧的列表
    video_frames = []
    for step, batch in enumerate(train_dataloader):
        # 假设 batch 是一个包含多个视频帧的列表
        video_frames.extend(batch)
    #print("第一个循环结束")

    # 对每个视频帧进行处理
    for frame in video_frames:  # video_frames是包含所有视频帧的列表
        process_video_frame(frame)
    #print('修改1完成')
    '''
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {dataloader_length}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    epoch_loss = 0.0
    step_per_epoch = 1
    
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        with open(os.path.join(args.output_dir, 'logloglog.txt'), 'w' if epoch == 0 else 'a') as f:
            f.write(f'{epoch} {epoch_loss} {step_per_epoch} {epoch_loss / step_per_epoch}\n')
        epoch_loss = 0.0
        step_per_epoch = 0
        
        for step, batch in enumerate(train_dataloader):
            #print(f'train by labeled data: step={step}, image = {batch[0].shape}, label = {batch[1].shape}, noise = {batch[2].shape}')
            with accelerator.accumulate(unet):
                #image, text_embeddings, noise = batch
                device = accelerator.device
                image, text_embeddings, depth = batch

                if args.mixed_precision == "fp16":
                    image = image.half()
                    text_embeddings = text_embeddings.half()
                    depth = depth.half()

                image = image[0] # b, 2, 3, 512, 512
                frame_1 = image[0][0]
                frame_2 = image[0][1]
                image = torch.cat([image, image], dim=1)
                image = image.reshape(-1, image.shape[2], image.shape[3], image.shape[4]).cuda()# b*2*2, 3, 512, 512
                

                text_embeddings = text_embeddings[0].cuda() # b 20 77 768
                text_embeddings = text_embeddings[:, 3, :, :] # Select a description b 77 768
                text_embeddings = torch.cat([text_embeddings]*4, dim=0).cuda() # b*2*2, 77, 768
                #print(image.shape, text_embeddings.shape)

                depth = depth[0].repeat(1, 1, 3, 1, 1).view(-1, 3, 512, 512)
                depth = torch.cat([depth, depth], dim=0)

                latents = vae.encode(image).latent_dist.sample()
                latents = latents * vae.config.scaling_factor  # 4, 4, 64, 64

                noise = torch.randn(
                    latents.shape, device=latents.device
                )

                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                if args.mixed_precision == "fp16":
                    noise = noise.half()
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(int(0.9 * noise_scheduler.config.num_train_timesteps), noise_scheduler.config.num_train_timesteps, (bsz//2,), device=latents.device).reshape(-1, 1)
                timesteps = torch.concatenate([timesteps, timesteps], dim=1).reshape(-1)
                timesteps = timesteps.long()


                # use pretrained prompt_learner
                output_prefix_token = loaded_prompt_learner(image, clip_model, clip_processor)

                '''if(random.random() < args.prompt_threshold):
                    encoder_hidden_states = output_prefix_token
                else:
                    encoder_hidden_states = text_embeddings'''
            
                encoder_hidden_states = torch.cat([output_prefix_token, text_embeddings], dim = 1)
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                # Predict the noise residual and compute loss
                assert not torch.isnan(noisy_latents).any()
                assert not torch.isinf(noisy_latents).any()

                controlnet_image = depth
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                #model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                # if torch.isinf(model_pred).any():
                #     raise ValueError("预测模型出现inf")
                # if torch.isinf(target).any():
                #     raise ValueError("target出现inf")
                # if torch.isnan(model_pred).any():
                #     raise ValueError("预测模型出现nan")
                # if torch.isnan(target).any():
                #     raise ValueError("target出现nan")


                # reconstruct_loss
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                    
                if torch.isinf(loss).any():
                    raise ValueError("loss出现inf")
                if torch.isnan(loss).any():
                    raise ValueError("loss出现nan")

                features = hook.get_features() #(2, 32 * 4, 64, 64)
                # features = [torch.randn((4, 32, 64, 64)), torch.randn((4, 64, 32, 32))]
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                ratio = args.ratio
                old_loss = loss
                our_loss = 0.
                for ind, feature in enumerate(features): #(32 * 4, 64, 64)
                    #print(ind, feature.shape)
                    split_features = torch.chunk(feature, 4, dim=0) #(4 * (32, 64, 64))
                    f1_t1 = split_features[0]
                    f2_t1 = split_features[1] # b//4 c h w
                    f2_t2 = split_features[2]
                    f1_t2 = split_features[3]
                    
                    def reshape_and_permute(f):
                        c, h, w = f.shape
                        return f.permute(1, 2, 0).reshape(-1, c)
                    tmp1 = reshape_and_permute(f1_t1)
                    tmp2 = reshape_and_permute(f2_t1)
                    tmp3 = reshape_and_permute(f1_t2)
                    tmp4 = reshape_and_permute(f2_t2)
                    sim_12 = cos(tmp1, tmp2)
                    sim_34 = cos(tmp3, tmp4)
                    cur_loss = ratio * ((sim_12 - sim_34) ** 2).mean()
                    our_loss += cur_loss
                    #print("our_loss = ", our_loss)
                    #print(type(our_loss), type(cur_loss))
                hook.reset()
                
                del features

                #final_loss = old_loss + our_loss
                final_loss = old_loss
                # if our_loss <= 0.0075:
                #     final_loss += our_loss
                final_loss = old_loss + our_loss
                    
                if torch.isinf(our_loss).any():
                    raise ValueError("our_loss出现inf")
                if torch.isnan(our_loss).any():
                    raise ValueError("our_loss出现nan")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(final_loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(final_loss)

                if accelerator.sync_gradients:
                    params_to_clip = all_params
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                epoch_loss += train_loss
                step_per_epoch += 1
                
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        unwrapped_unet = unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )
                        prompt_learner = unwrap_model(loaded_prompt_learner)
                        prompt_learner.save_model("prefix_token_model.pth")

                        logger.info(f"Saved state to {save_path}")

            #logs = {"step_loss": old_loss.detach().item(), "our_loss": our_loss.detach().item(), "total_loss": final_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            logs = {"step_loss": old_loss.detach().item(), "our_loss":our_loss.detach().item(), "total_loss": final_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )

                # create pipeline
                '''pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    cache_dir=args.cache_dir
                )'''
                pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    controlnet=controlnet,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=torch.float16,
                    cache_dir=args.cache_dir
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device)
                if args.seed is not None:
                    generator = generator.manual_seed(args.seed)
                images = []
                with torch.cuda.amp.autocast():
                    for _ in range(args.num_validation_images):
                        images.append(
                            pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0]
                        )

                if False:
                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        #controlnet = controlnet.to(torch.float32)

        #controlnet = unwrap_model(controlnet)
        #controlnet.save_pretrained(args.output_dir)

        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionControlNetPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )
        prompt_learner = unwrap_model(loaded_prompt_learner)
        prompt_learner.save_model("prefix_token_model.pth")

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        # Final inference
        # Load previous pipeline
        if False and args.validation_prompt is not None:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                cache_dir=args.cache_dir
            )
            pipeline = pipeline.to(accelerator.device)

            # load attention processors
            pipeline.load_lora_weights(args.output_dir)

            # run inference
            generator = torch.Generator(device=accelerator.device)
            if args.seed is not None:
                generator = generator.manual_seed(args.seed)
            images = []
            with torch.cuda.amp.autocast():
                for _ in range(args.num_validation_images):
                    images.append(
                        pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0]
                    )

            for tracker in accelerator.trackers:
                if len(images) != 0:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "test": [
                                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
