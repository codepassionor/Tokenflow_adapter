import torch
from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms as transforms
from einops import rearrange
import json
import os
import cv2
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import webdataset as wds
'''
#[SYM]: different API with `MSRVTTLocalDataset`.

class RandomImageDataset(Dataset):
    def __init__(self, num_samples, num_channels=4, height=64, width=64):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            num_channels (int): Number of channels in the images.
            height (int): The height of the images.
            width (int): The width of the images.
        """
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.height = height
        self.width = width

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random image
        noise = torch.randn(self.num_channels, self.height, self.width)
        target = torch.randn(self.num_channels, self.height, self.width)
        text = torch.randn(77, 768)
        return noise, target, text

def get_data_loader(batch_size):
    # Instantiate the dataset
    num_samples = 1000  # total number of samples in the dataset
    dataset = RandomImageDataset(num_samples)

    # Create a DataLoader
 to batch the data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader
'''

class MSRVTTLocalDataset(Dataset):
    def __init__(self, dataset_src: str, train: bool, transformer, video_subfolder='video'):
        self.train = train
        self.dataset_src = dataset_src
        self.transformer = transformer
        data_info_src = os.path.join(dataset_src, 'MSRVTT_data.json')
        with open(data_info_src, 'r') as f:
            data_info = json.loads(f.read())
        self.data_info = data_info

        video_subfolder = os.path.join(dataset_src, video_subfolder)

        self.dataset_src = [
            os.path.join(video_subfolder, video['video_id']) + '.mp4'
            for video in data_info['videos'] if (video['split'] == 'train') == (train)
        ]

        self.num_samples = len(self.dataset_src)

        # prompts[video_id] is the list of the prompts of video_id
        prompts = {}
        for sentence in data_info['sentences']:
            if prompts.get(sentence["video_id"]):
                prompts[sentence["video_id"]].append(sentence["caption"])
            else:
                prompts[sentence["video_id"]] = [sentence["caption"]]


        self.prompts = [
            prompts[video["video_id"]]
            for video in tqdm(data_info['videos']) if (video['split'] == 'train') == (train)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        video_src_path = self.dataset_src[idx]
        video = cv2.VideoCapture(video_src_path)
        # [SYM]: Only load the first frame for a certain video.
        success, frame1 = video.read()
        if not success:
            raise RuntimeError(f"Can not load the frames of {video_src_path}")
        success, frame2 = video.read()
        if not success:
            raise RuntimeError(f"Can not load the frames of {video_src_path}")
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        pil_image1 = Image.fromarray(frame1_rgb)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        pil_image2 = Image.fromarray(frame2_rgb)
        video.release()
        # [SYM]: Only use the first sentance for a certain video.
        prompt = self.prompts[idx][0]
        tensor_image1 = self.transformer(pil_image1)
        tensor_image2 = self.transformer(pil_image2)
        tensor_image1 = tensor_image1
        tensor_image2 = tensor_image2
        stacked_images = torch.concatenate([tensor_image1,tensor_image2,tensor_image1,tensor_image2], dim=0)
        return stacked_images, prompt


if __name__ == '__main__':
    # pass
    # print('start')
    # tokenizer = CLIPTokenizer.from_pretrained(
    #     'runwayml/stable-diffusion-v1-5',
    #     subfolder='tokenizer'
    # )
    # text_encoder = CLIPTextModel.from_pretrained(
    #     'runwayml/stable-diffusion-v1-5',
    #     subfolder='text_encoder'
    # )
    # text_encoder.to('cuda')
    # vae = AutoencoderKL.from_pretrained(
    #     'runwayml/stable-diffusion-v1-5', subfolder="vae"
    # )
    # vae.to('cuda')
    # print('pretrained model loaded.')

    dataset = MSRVTTLocalDataset(
        dataset_src="Tokenflow_adapter/vid_edit_friendly_t2i/MSRVTT",
        train=True,
        transformer = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    )
    # print('dateset built')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # # dataset, dataloader = get_data_loader(4)
    for images, text in dataloader:
        print(images[0].shape,images[1].shape)
        print(text)

