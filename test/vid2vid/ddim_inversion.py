from typing import Union, Tuple, Optional
import numpy as np
import cv2

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from torchvision import transforms as tvt
import decord
from decord import VideoReader
from torchvision import transforms


import cv2
import numpy as np
from PIL import Image

import torch.nn.functional as F

def bilateral_filter_1d_vectorized(input, kernel_size, sigma_spatial, sigma_intensity):
    """
    Apply a vectorized bilateral filter on 1D input data.

    Args:
    - input (torch.Tensor): The input tensor of shape (batch_size, channels, width).
    - kernel_size (int): The size of the filter kernel.
    - sigma_spatial (float): The spatial distance standard deviation.
    - sigma_intensity (float): The intensity difference standard deviation.

    Returns:
    - filtered_output (torch.Tensor): The output after applying the bilateral filter.
    """
    half_kernel = kernel_size // 2
    batch_size, channels, width = input.shape

    # Pad the input to handle the borders
    padded_input = F.pad(input, (half_kernel, half_kernel), mode='reflect')

    # Create the spatial Gaussian kernel (same for all data points)
    spatial_gaussian = torch.exp(-torch.pow(torch.arange(-half_kernel, half_kernel + 1, device=input.device, dtype=input.dtype), 2) / (2 * sigma_spatial ** 2))

    # Create a sliding window of the local regions for each element in the input
    local_regions = padded_input.unfold(2, kernel_size, 1)  # (batch_size, channels, width, kernel_size)

    # Compute the intensity differences (vectorized for all elements)
    center_pixel = input.unsqueeze(-1)  # Shape: (batch_size, channels, width, 1)
    intensity_gaussian = torch.exp(-torch.pow(local_regions - center_pixel, 2) / (2 * sigma_intensity ** 2))

    # Combine spatial and intensity components
    bilateral_kernel = spatial_gaussian.view(1, 1, 1, -1) * intensity_gaussian

    # Normalize the kernel
    bilateral_kernel = bilateral_kernel / bilateral_kernel.sum(dim=-1, keepdim=True)

    # Apply the kernel to the local regions
    filtered_output = (bilateral_kernel * local_regions).sum(dim=-1)

    return filtered_output


def write_images_to_mp4(image_list, output_filename, fps=30):
    # Convert the first image to a NumPy array to get the size (height, width)
    first_image = np.array(image_list[0])
    height, width, layers = first_image.shape

    # Initialize the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for img in image_list:
        # Convert the PIL image to a NumPy array
        img_np = np.array(img)

        # Convert RGB (PIL format) to BGR (OpenCV format)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Write the frame to the video
        video.write(img_np)

    # Release the VideoWriter
    video.release()

NUM = 5


def resize_keep_aspect_ratio(img, target):
    assert isinstance(target, int)
    h, w = img.size[1], img.size[0]
    if h >= w:
        wpercent = (target / float(w))
        hsize = int((float(h) * float(wpercent)))
        img = img.resize((target, hsize), Image.Resampling.LANCZOS)
        return img.crop((0, 0, target, target))
    else:
        hpercent = (target / float(h))
        wsize = int((float(w) * float(hpercent)))
        img = img.resize((wsize, target), Image.Resampling.LANCZOS)
        return img.crop((0, 0, target, target))


def load_image(img: Union[str, Image.Image] , target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    if isinstance(img, str):
        pil_img = Image.open(img).convert('RGB')

    if target_size is not None:
        #if isinstance(target_size, int):
        #    target_size = (target_size, target_size)
        #pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        pil_img = resize_keep_aspect_ratio(pil_img, target_size)   # 此处报错，pil_img未定义

    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents

def latents_to_img(self, latents, return_type='np'):
    latents = 1 / 0.18215 * latents.detach()
    image = self.model.vae.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    return image


class BFHooker:
    def __init__(self, model):
        self.model = model
        self.features = []
        self.hooks = []
        self.register_hook()

    def hook_function(self, module, input, output):
        #return torch.flip(output, [0]) # this is a extreme case, but still work in low CFG case

        # B, N, d
        output = output.permute(1, 2, 0)
        # N，d, B
        output = bilateral_filter_1d_vectorized(output, 5, 1, 1)
        output = output.permute(2, 0, 1)

        return output

    def register_hook(self):
        #print(self.model._modules)
        #res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
        res_dict = {3: [0, 1]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
        unet = self.model
        for res in res_dict:
            for block in res_dict[res]:
                module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
                self.hooks.append(module.register_forward_hook(self.hook_function))

    def get_features(self, input_data):
        _ = self.model(input_data)
        return self.features

    def reset(self):
        self.features = []

    def remove_hook(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


@torch.no_grad()
def ddim_inversion_video(video_path: str, num_steps: int = 50, verify: Optional[bool] = False, prompt="", negative_prompt="", guidance_scale=7.5, target_size=512, pretrain_path='runwayml/stable-diffusion-v1-5') -> torch.Tensor:
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    inverse_scheduler = DDIMInverseScheduler.from_pretrained(pretrain_path, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(pretrain_path,
                                                   scheduler=inverse_scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=dtype)

    vr = VideoReader(video_path)
    cnt = NUM
    i = 0
    tensor_list = []
    transform = transforms.ToTensor()
    for frame in vr:
        frame = frame.asnumpy()
        tensor_list.append(transform(frame))
        i += 1
        if i >= cnt:
            break

    # Stack the tensors to create a batch
    image_batch = torch.stack(tensor_list)

    hooker = BFHooker(pipe.unet)

    pipe.to(device)
    vae = pipe.vae

    #input_img = load_image(imgname, target_size=target_size).to(device=device, dtype=dtype)
    print(image_batch.shape)
    latents = img_to_latents(image_batch.to(device=device, dtype=dtype), vae)

    del image_batch

    # 使用 pipe 进行去噪，并保存噪声
    inv_latents = pipe(prompt=[prompt]*NUM, negative_prompt=[negative_prompt]*NUM, guidance_scale=guidance_scale,
                              width=512, height=512,
                              output_type='latent', return_dict=False,
                              num_inference_steps=num_steps, latents=latents, return_noise=True)[0]

    del latents


    # verify
    verify = True
    if verify:
        #hooker.remove_hook()
        pipe.scheduler = DDIMScheduler.from_pretrained(pretrain_path, subfolder='scheduler')
        out = pipe(prompt=[prompt]*NUM, negative_prompt=[negative_prompt]*NUM, guidance_scale=guidance_scale,
                     num_inference_steps=num_steps, latents=inv_latents)
        images = out.images
        write_images_to_mp4(images, 'ori_reverse_bilater_4_step50_hook.mp4', fps=10)
        #from IPython import embed; embed()
        #print(image.shape)
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(tvt.ToPILImage()(input_img[0]))
        #ax[1].imshow(image.images[0])
        ##plt.show()
        #plt.savefig('hack_test_out5.png')
        #cv2.imwrite('hack_test_in.png', np.transpose(input_img.cpu().numpy()[0], (1, 2, 0))[:, :, ::-1])

    return inv_latents


if __name__ == '__main__':
    filename = '/data/workspace/prompt-to-prompt/wolf.mp4'
    #ret = ddim_inversion_video('./pokemon.jpg', num_steps=50, prompt='a blue pokemon, cute, character', guidance_scale=1.0, verify=True)
    #ret = ddim_inversion_video(filename, num_steps=100, prompt='wolf', guidance_scale=7.5, verify=True)
    ret = ddim_inversion_video(filename, num_steps=50, prompt='wolf', guidance_scale=1.0, verify=True)
    #print(ret)
    #print(ret.shape)
