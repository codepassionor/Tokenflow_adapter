import os
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
import torchvision
from torchvision.io import write_video
from pathlib import Path
import torchvision.transforms as transforms
from itertools import islice
import tempfile
from torch.utils.data import Dataset, DataLoader
from pytorch_fid import fid_score
import lpips
import cv2
from PIL import Image
import numpy as np
import glob
from torchvision.models import Inception_V3_Weights
import json
import yaml
import time
import test_vid2vid_zero
import test_vid2vid_zero_lora
from omegaconf import OmegaConf



def calc_lpips(image_paths):

    images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
    images = [torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0 for img in images]

    lpips_scores = []
    for i in range(len(images) - 1):
        score = lpips_model(images[i], images[i + 1])
        lpips_scores.append(score.item())

    avg_lpips = sum(lpips_scores) / len(lpips_scores)
    return avg_lpips

def get_clip_score(image_path,text):
    score = []
    for path in image_path:
        image = Image.open(path)
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        score.append(logits_per_image)
    clip_score = (sum(score)/len(score)).item()
    return clip_score

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_folder, f"frame{count:04d}.png"), frame)
            count += 1
        else:
            break
    cap.release()

if __name__ == "__main__":
    inception_model = torchvision.models.inception_v3(pretrained=True)
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("/root/autodl-tmp/cache_huggingface/huggingface/models--openai--clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("/root/autodl-tmp/cache_huggingface/huggingface/models--openai--clip-vit-base-patch32")
    lpips_model = lpips.LPIPS(net='alex')

    CLIP_score = []
    FID_score = []
    IPIPS_score = []
    CLIP_score_lora = []
    FID_score_lora = []
    IPIPS_score_lora = []
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=f"./configs/skateboard-man.yaml")
    args = parser.parse_args()
    test_vid2vid_zero.main(**OmegaConf.load(args.config))
    test_vid2vid_zero_lora.main(**OmegaConf.load(args.config))
    '''
    yaml_files = glob.glob(f'./configs/*.yaml')
    num = 0
    for file_name in yaml_files:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default=file_name)
        parser.add_argument('--begin', type=int, default=0)
        parser.add_argument('--end', type=int, default=300)
        #parser.add_argument("--config", type=str, default=f"./configs/config_new_{i:02d}.yaml")
        args = parser.parse_args()
        test_vid2vid_zero.main(**OmegaConf.load(args.config), lora_begin=args.begin, lora_end=args.end)
        test_vid2vid_zero_lora.main(**OmegaConf.load(args.config), lora_begin=args.begin, lora_end=args.end)

        # test
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        for prompt in config["validation_data"]["prompts"]:
            generated_images_folder = f'{args.begin}/' + config["output_dir"] + "/base/sample"
            extract_frames(generated_images_folder + "/" + prompt + ".mp4", generated_images_folder)
            image_paths = glob.glob(generated_images_folder + "/*.png")

            generated_images_folder_lora = f'{args.begin}/' + config["output_dir"] + "/lora/sample"
            extract_frames(generated_images_folder_lora + "/" + prompt + ".mp4", generated_images_folder_lora)
            image_paths_lora = glob.glob(generated_images_folder_lora + "/*.png")

            # CLIP
            clip_score = get_clip_score(image_paths, prompt)
            clip_score_lora = get_clip_score(image_paths_lora, prompt)
            print('CLIP_Score:', clip_score)
            print('CLIP_Score with lora:', clip_score_lora)

            # FID
            parts = file_name.split('/')
            name = parts[-1]
            real_images_folder = "data/data_example/" + name[:-5]
            fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                    batch_size=16, device='cuda', dims=2048)
            fid_value_lora = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder_lora],
                                                    batch_size=16, device='cuda', dims=2048)
            print('FID value:', fid_value)
            print('FID value with lora:', fid_value_lora)


            # LPIPS
            lpips_similarity = calc_lpips(image_paths)
            lpips_similarity_lora = calc_lpips(image_paths_lora)
            print("Average LPIPS similarity:", lpips_similarity)
            print("Average LPIPS similarity with lora:", lpips_similarity_lora)
            CLIP_score.append(clip_score)
            FID_score.append(fid_value)
            IPIPS_score.append(lpips_similarity)
            CLIP_score_lora.append(clip_score_lora)
            FID_score_lora.append(fid_value_lora)
            IPIPS_score_lora.append(lpips_similarity_lora)

    with open(f'output{args.begin}.txt', 'w') as file:

        for element in CLIP_score:
            file.write(str(element)+'\n')
        file.write('\n')
        for element in FID_score:
            file.write(str(element)+'\n')
        file.write('\n')
        for element in IPIPS_score:
            file.write(str(element)+'\n')
        file.write('\n')

        for element in CLIP_score_lora:
            file.write(str(element)+'\n')
        file.write('\n')
        for element in FID_score_lora:
            file.write(str(element)+'\n')
        file.write('\n')
        for element in IPIPS_score_lora:
            file.write(str(element)+'\n')
        file.write('\n')