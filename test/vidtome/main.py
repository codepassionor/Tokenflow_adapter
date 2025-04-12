from invert import Inverter
from generate import Generator
from utils import load_config, init_model, seed_everything, get_frame_ids, load_config_example
import os
import glob
from PIL import Image
import torch
import torchvision
from transformers import CLIPProcessor, CLIPModel
import run_vidtome
from pathlib import Path
import argparse
import lpips
import json
import yaml
from pytorch_fid import fid_score
import numpy as np
import cv2
from ddim_inversion import BFHooker

def calc_lpips(image_paths):
    images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
    images = [torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0 for img in images]

    lpips_scores = []
    for i in range(len(images) - 1):
        score = lpips_model(images[i], images[i + 1])
        lpips_scores.append(score.item())

    avg_lpips = sum(lpips_scores) / len(lpips_scores)
    return avg_lpips

def get_clip_score(image_path, text):
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

    clip_model = CLIPModel.from_pretrained("/root/autodl-tmp/cache_huggingface/huggingface/openai/clip-vit-base-patch32/")
    processor = CLIPProcessor.from_pretrained("/root/autodl-tmp/cache_huggingface/huggingface/openai/clip-vit-base-patch32/")
    lpips_model = lpips.LPIPS(net='alex')

    CLIP_score = []
    FID_score = []
    IPIPS_score = []
    CLIP_score_lora = []
    FID_score_lora = []
    IPIPS_score_lora = []
    
    for i in range(6):
        config, lora_begin, lora_end = load_config(i)
        pipe, scheduler, model_key = init_model(
            config.device, config.sd_version, config.model_key, config.generation.control, config.float_precision)
        config.model_key = model_key
        seed_everything(config.seed)
        
        for prompt in config['generation']['prompt']:
            for key, value in prompt.items():
                prompt_id = key
                prompt_word = value
            print("Start inversion!")
            hooker = BFHooker(pipe.unet)
            inversion = Inverter(pipe, scheduler, config)
            inversion.force = True
            inversion(config.input_path, config.inversion.save_path, lora_begin, lora_end, use_prefixtoken=False)
            
            print("Start generation!")
            #hooker.remove_hook()
            generator = Generator(pipe, scheduler, config, prompt)
            frame_ids = get_frame_ids(
                config.generation.frame_range, config.generation.frame_ids)
            generator(config.input_path, config.generation.latents_path,
                    f'{lora_begin}/' + config.generation.output_path + '/' + prompt_id + "/base", frame_ids=frame_ids, lora_begin=lora_begin, lora_end=lora_end, use_prefixtoken=False)
            
            print("Start inversion!")
            pipe.load_lora_weights("checkpoint/token/prefix8/pytorch_lora_weights.safetensors")
            pipe.fuse_lora(lora_scale=0.5)
            hooker = BFHooker(pipe.unet)
            inversion = Inverter(pipe, scheduler, config)
            inversion.withlora = True
            #inversion.force = True
            inversion(config.input_path, config.inversion.save_path, lora_begin, lora_end, use_prefixtoken=False)
            
            print("Start generation!")
            #hooker.remove_hook()
            generator = Generator(pipe, scheduler, config, prompt)
            generator.withlora = True
            frame_ids = get_frame_ids(
                config.generation.frame_range, config.generation.frame_ids)
            generator(config.input_path, config.generation.latents_path,
                    f'{lora_begin}/' + config.generation.output_path + '/' + prompt_id + "/lora", frame_ids=frame_ids, lora_begin=lora_begin, lora_end=lora_end, use_prefixtoken=True)

            
            # test
            generated_images_folder = f'{lora_begin}/' + config["work_dir"] + '/' + prompt_id + "/base" + "/" + prompt_id + "/frames"
            image_paths = glob.glob(f'{lora_begin}/' + config["work_dir"] + '/' + prompt_id + "/base" + "/" + prompt_id + "/frames" + "/*.png")
            generated_images_folder_lora = f'{lora_begin}/' + config["work_dir"] + '/' + prompt_id + "/lora" + "/" + prompt_id + "/frames"
            image_paths_lora = glob.glob(f'{lora_begin}/' + config["work_dir"] + '/' + prompt_id + "/lora" + "/" + prompt_id + "/frames" + "/*.png")
            real_images_folder = f"data/" + config["work_dir"]


            # CLIP
            clip_score = get_clip_score(image_paths, prompt_word)
            clip_score_lora = get_clip_score(image_paths_lora, prompt_word)
            print('CLIP_Score:', clip_score)
            print('CLIP_Score with lora:', clip_score_lora)

            # FID
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



    with open(f'output{lora_begin}.txt', 'w') as file:
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


