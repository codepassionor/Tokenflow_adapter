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
    FID_score = []
    IPIPS_score = []
    num = 0

    for video, metadata in dataset:
        for i in range(0,31):
            with open(f"textset/{str(num).zfill(2)}/{str(i).zfill(5)}.txt", 'a') as file:
                file.write(metadata["caption_0"]+'\n')
        num+=1
        if num >= 10:
            break