import torch
import torchvision.transforms as transforms
import webdataset as wds
from io import BytesIO
from PIL import Image

dataset = wds.WebDataset("data/msrvtt.tar").decode(
    wds.handle_extension("input.ppm", transforms.Compose([
            lambda x : Image.open(BytesIO(x)),
            transforms.ToTensor()
        ])),
)


for i, sample in enumerate(dataset):
    for key, value in sample.items():
        print(key, type(value), repr(value)[:100])
    print()
    if i >= 5: break
