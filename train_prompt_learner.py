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
import torchvision
from torchvision import transforms
import webdataset as wds
from vid_edit_dataset import get_data_loader
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import transformers
from transformers import CLIPProcessor, CLIPModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

class PrefixToken(torch.nn.Module):
    def __init__(self, share_token_num):
        super().__init__()

        self.projection = torch.nn.Linear(768, 768)
        self.share_token = torch.nn.Parameter(torch.randn(share_token_num, 768))

        self.projection.requires_grad_(True)
        self.share_token.requires_grad_(True)

        # Initialize the weights of the linear layer
        self._initialize_weights()

    def _initialize_weights(self):
        # Using Xavier initialization (Glorot) for the weights
        torch.nn.init.xavier_uniform_(self.projection.weight)

        # Initialize the bias with zeros
        if self.projection.bias is not None:
            torch.nn.init.zeros_(self.projection.bias)

        # Initialize the shared token parameter with a normal distribution
        torch.nn.init.normal_(self.share_token, mean=0.0, std=0.02)

    def forward(self, image, clip_model, clip_processor, sub=1):
        inputs = clip_processor(images=image, return_tensors="pt", do_rescale=False)
        vision_outputs = clip_model.vision_model(
            pixel_values=inputs['pixel_values'].to('cuda'),
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = vision_outputs['last_hidden_state']  # B, 50, dim
        hidden = torch.cat([hidden[:, :1, :], hidden[:, 1::sub, : ]], dim=1)
        hidden = hidden.to(self.projection.weight)

        unshare_token = self.projection(hidden)
        prefix_token = torch.cat([self.share_token.unsqueeze(0).expand(hidden.size(0), -1, -1), unshare_token], dim=1)

        return prefix_token

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path, device):
        self.load_state_dict(torch.load(file_path, map_location=device))


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_src",
        type=str,
        default="/root/autodl-tmp/dataset/wd_MSRVTT",
        help=("File of `.tar` which is webdataset file. e.g.:/root/autodl-tmp/data/msrvtt.tar")
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/root/autodl-tmp/dataset",
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
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
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
        default=1e-4,
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
        default=500,
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

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    return args

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




DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def main():
    # Video2webdatset()
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models.

    clip_model = CLIPModel.from_pretrained("/root/autodl-tmp/dataset/openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("/root/autodl-tmp/dataset/openai/clip-vit-base-patch32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)
    prompt_learner = PrefixToken(4).cuda()

    # Save the module
    save_path = "prefix_token_model.pth"
    prompt_learner.save(save_path)
    print(f"Model saved to {save_path}")

    # Load the module
    loaded_prompt_learner = PrefixToken(4)
    loaded_prompt_learner.load(save_path, device)
    loaded_prompt_learner.to(device)
    print("Model loaded successfully")


    if args.mixed_precision == "fp16":
        # only upcast trainable parameters into fp32
        cast_training_params(loaded_prompt_learner, dtype=torch.float32)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.Adam(loaded_prompt_learner.parameters(), lr=1e-3)

    # TODO: [SYM]:
    # fingerprint used by the cache for the other processes to load the result
    # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401

    # DataLoaders creation:
    # [SYM]: Can not use batch in webdataset, which is inconsistant with accelerator.
    # @See: https://github.com/huggingface/accelerate/issues/1370
    # @See: https://github.com/huggingface/accelerate/issues?q=is%3Aissue+is%3Aopen+webdataset

    train_dataloader = get_data_loader(args.train_batch_size, args.dataloader_num_workers)
    dataloader_length = 10000 // args.train_batch_size

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(dataloader_length / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # HOOK

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(dataloader_length / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    losses = []
    for epoch in tqdm(range(10)):

        for step, batch in enumerate(train_dataloader):
            
            image, text_embeddings = batch
            # Zero gradients
            optimizer.zero_grad()
            image_t = image[0][0][0]
            prefix_tokens = loaded_prompt_learner(image_t, clip_model, clip_processor, sub=2) # subsample

            ## sd inference, concat the prefix token with the text embedding, note that the maximum tokens is set to 77, you might need to set larger value in clip config
            # final_text_embedding = torch.cat([prefix_tokens, text_embdding], dim=1)
            # stable_diffusion(xxxx, final_text_embedding)
            ##

            ## the following is loss is not for usage, just for backward example

            loss = (prefix_tokens).mean()

            losses.append(loss.cpu().item())

            loss.backward()
            optimizer.step()

    save_path = "prefix_token_model.pth"
    prompt_learner.save(save_path)
    print(f"Model saved to {save_path}")



if __name__ == "__main__":
    main()