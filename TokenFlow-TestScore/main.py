import os
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
import torchvision
from torchvision.io import write_video
from pathlib import Path
from util import *
import torchvision.transforms as T
import webdataset as wds
from itertools import islice
import tempfile
from torch.utils.data import DataLoader
import preprocess
import run_tokenflow_pnp
from pytorch_fid import fid_score
import lpips
from PIL import Image
import numpy as np
import glob
from torchvision.models import Inception_V3_Weights


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


if __name__ == "__main__":
    device = 'cuda'
    sharedurl = "dataset/00000.tar"
    dataset = (
    wds.WebDataset(sharedurl)
    .decode("torchrgb")
    .to_tuple("mp4", "json")
    )
    loader = DataLoader(dataset, num_workers=4, batch_size=32)
    inception_model = torchvision.models.inception_v3(pretrained=True)
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    lpips_model = lpips.LPIPS(net='alex')

    FID_score = []
    IPIPS_score = []
    num = 0

    for video, metadata in dataset:
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_path', type=str,
                            default=metadata["video_id"]) 
        parser.add_argument('--H', type=int, default=240, 
                            help='for non-square videos, we recommand using 672 x 384 or 384 x 672, aspect ratio 1.75')
        parser.add_argument('--W', type=int, default=320, 
                            help='for non-square videos, we recommand using 672 x 384 or 384 x 672, aspect ratio 1.75')
        parser.add_argument('--save_dir', type=str, default='latents')
        parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1', 'ControlNet', 'depth'],
                            help="stable diffusion version")
        parser.add_argument('--steps', type=int, default=500)
        parser.add_argument('--batch_size', type=int, default=40)
        parser.add_argument('--save_steps', type=int, default=50)
        parser.add_argument('--n_frames', type=int, default=40)
        parser.add_argument('--inversion_prompt', type=str, default=metadata["caption_0"])
        opt = parser.parse_args()
        video_path = opt.data_path
        save_video_frames(video, video_path, img_size=(opt.W, opt.H))
        opt.data_path = os.path.join('data', Path(video_path).stem)
        #preprocess.prep(opt)

        parser.add_argument('--config_path', type=str, default='configs/config_pnp.yaml')
        opt = parser.parse_args()
        with open(opt.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        config['data_path'] = 'data/' + metadata["video_id"]
        config['prompt'] = metadata["caption_0"]
        config["output_path"] = os.path.join("tokenflow-results" + f'_pnp_SD_{config["sd_version"]}',
                                                Path(config["data_path"]).stem,
                                                config["prompt"][:240],
                                                f'attn_{config["pnp_attn_t"]}_f_{config["pnp_f_t"]}',
                                                f'batch_size_{str(config["batch_size"])}',
                                                str(config["n_timesteps"]),
        )
        os.makedirs(config["output_path"], exist_ok=True)
        assert os.path.exists(config["data_path"]), "Data path does not exist"
        with open(opt.config_path, "w") as file:
            yaml.dump(config, file)
        with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
            yaml.dump(config, f)
        #run_tokenflow_pnp.run(config)


        # test
        real_images_folder = config["output_path"] + "/vae_recon"
        generated_images_folder = config["output_path"] + "/img_ode"
        image_paths = glob.glob(config["output_path"] + "/img_ode" + "/*.png")

        # CLIP
        clip_score = get_clip_score(image_paths, metadata["caption_0"])
        print('FID value:', clip_score)
        with open('result.txt', 'a') as file:
            file.write(f"CLIP_Score:{clip_score:.6f}"+'\n')

        # FID
        fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                batch_size=16, device='cuda', dims=2048)
        FID_score.append(fid_value)
        print('FID value:', fid_value)
        with open('result.txt', 'a') as file:
            file.write(f"FID value:{fid_value:.6f}"+'\n')

        # LPIPS
        lpips_similarity = calc_lpips(image_paths)
        print("Average LPIPS similarity:", lpips_similarity)
        IPIPS_score.append(lpips_similarity)
        with open('result.txt', 'a') as file:
            file.write(f"Average LPIPS similarity: {lpips_similarity:.6f}" +'\n')

        num += 1
        if num >= 10:
            break
