import torch
from model import Model
from model_lora import Model_lora
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from pytorch_fid import fid_score
import lpips
import torchvision
import os
import cv2
import glob
import json
import argparse

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
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32/")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32/")
    lpips_model = lpips.LPIPS(net='alex')
    model = Model(device = "cuda", dtype = torch.float16)
    model_lora = Model_lora(device = "cuda", dtype = torch.float16)
    CLIP_score = []
    IPIPS_score = []
    CLIP_score_lora = []
    IPIPS_score_lora = []
    model_name = "runwayml/stable-diffusion-v1-5"

    for i in range(6):
        parser = argparse.ArgumentParser()
        parser.add_argument('--begin', type=int, default=0)
        parser.add_argument('--end', type=int, default=300)
        opt = parser.parse_args()

        with open(f'data/test_{i}.json') as f:
            metadata = json.load(f)
        prompt = metadata["caption_0"]
        params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}

        os.makedirs(f"./output{opt.begin}", exist_ok=True)
        out_path, fps = f"./output{opt.begin}/text2video_{prompt.replace(' ','_')}.mp4", 4
        model.process_text2video(prompt, model_name=model_name, fps = fps, path = out_path, **params)
        os.makedirs(f"./output_lora{opt.begin}", exist_ok=True)
        out_path_lora, fps = f"./output_lora{opt.begin}/text2video_{prompt.replace(' ','_')}.mp4", 4
        model_lora.process_text2video(prompt, model_name=model_name, fps = fps, path = out_path_lora, lora_bigin=opt.begin, lora_end=opt.end,  **params)


        os.makedirs(f"./output{opt.begin}/text2video_{prompt.replace(' ','_')}", exist_ok=True)
        os.makedirs(f"./output_lora{opt.begin}/text2video_{prompt.replace(' ','_')}", exist_ok=True)
        generated_images_folder = f"./output{opt.begin}/text2video_{prompt.replace(' ','_')}"
        generated_images_folder_lora = f"./output_lora{opt.begin}/text2video_{prompt.replace(' ','_')}"
        extract_frames(out_path, generated_images_folder)
        extract_frames(out_path_lora, generated_images_folder_lora)

        image_paths = glob.glob(f"./output{opt.begin}/text2video_{prompt.replace(' ','_')}" + "/*.png")
        image_paths_lora = glob.glob(f"./output_lora{opt.begin}/text2video_{prompt.replace(' ','_')}" + "/*.png")
        clip_score = get_clip_score(image_paths, metadata["caption_0"])
        clip_score_lora = get_clip_score(image_paths_lora, metadata["caption_0"])
        '''
        fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                batch_size=16, device='cuda', dims=2048)
        fid_value_lora = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder_lora],
                                                batch_size=16, device='cuda', dims=2048)
        '''
        lpips_similarity = calc_lpips(image_paths)
        lpips_similarity_lora = calc_lpips(image_paths_lora)
        print(clip_score)
        print(clip_score_lora)
        print(lpips_similarity)
        print(lpips_similarity_lora)

        CLIP_score.append(clip_score)
        IPIPS_score.append(lpips_similarity)
        CLIP_score_lora.append(clip_score_lora)
        IPIPS_score_lora.append(lpips_similarity_lora)


    with open(f'output{opt.begin}.txt', 'w') as file:
        
        for element in CLIP_score:
            file.write(str(element)+'\n')
        file.write('\n')

        for element in IPIPS_score:
            file.write(str(element)+'\n')
        file.write('\n')
        

        for element in CLIP_score_lora:
            file.write(str(element)+'\n')
        file.write('\n')

        for element in IPIPS_score_lora:
            file.write(str(element)+'\n')
        file.write('\n')