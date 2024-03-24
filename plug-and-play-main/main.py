import os
import json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import yaml
import run_features_extraction
import run_pnp

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.folders = sorted(os.listdir(data_dir))
        self.folder_to_images = {}
        self.prompts = {}

        for folder in self.folders:
            folder_path = os.path.join(data_dir, folder)
            prompt_path = os.path.join(folder_path, 'prompt.json')

            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.prompts[folder] = json.load(f)

            images = sorted([img for img in os.listdir(folder_path) if img.endswith('.png')])
            self.folder_to_images[folder] = [os.path.join(folder_path, img) for img in images]

        self.index_map = []
        for folder, images in self.folder_to_images.items():
            for img in images:
                self.index_map.append((folder, img))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        folder, image_path = self.index_map[idx]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        prompt = self.prompts[folder]
        
        image_name = os.path.basename(image_path)
        
        return image, prompt, image_name
    
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = CustomDataset(data_dir='dataset', transform=transform)

for images, prompts, image_names in dataset:

    config_experiment_name = prompts['number']
    config_init_img = 'dataset/' + prompts['number'] + '/' + image_names
    run_features_extraction.main(config_experiment_name, config_init_img, image_names.rsplit('.', 1)[0])

    config_source_experiment_name = prompts['number']
    config_prompts = prompts['prompt']
    config_negative_prompt = prompts['negative_prompt']

    run_pnp.main(config_source_experiment_name, config_prompts, config_negative_prompt, image_names.rsplit('.', 1)[0])

