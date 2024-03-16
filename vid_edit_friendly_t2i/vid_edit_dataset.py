import torch
from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms as transforms

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

    # Create a DataLoader to batch the data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader

class MSRVTTLocalDataset(Dataset):
    def __init__(self):
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



if __name__ == '__main__':
    dataloader = get_data_loader(8)
    for noise, target, text in dataloader:
        print(noise.shape)
        print(target.shape)
        break
