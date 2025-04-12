import webdataset as wds
import cv2
import os
import json
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from itertools import combinations
import random

raw_msrvtt_src = 'dataset/MSRVTT'
video_subfolder = 'TrainValVideo'

'''def prepare_depth_maps(frame1, frame2, device='cuda'):d
    depth_maps = []
    model = depth_model(device)

    for frame in [frame1, frame2]:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth_map = model.predict(img)
        depth_maps.append(depth_map)

    return torch.cat(depth_maps).to(torch.float16).to(device)'''

@torch.no_grad()
def prepare_depth_maps(frame1, frame2, model_type='DPT_Large', device='cuda'):
    depth_maps = []
    midas = torch.hub.load("intel-isl/MiDaS", model_type, source='local')
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", source='local')

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    for frame in [frame1, frame2]:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        latent_h = img.shape[0] // 8
        latent_w = img.shape[1] // 8

        input_batch = transform(img).to(device)
        prediction = midas(input_batch)

        depth_map = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(latent_h, latent_w),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
        depth_maps.append(depth_map)

    return torch.cat(depth_maps).to(torch.float16).to(device)

def encode_prompt(prompt_batch, text_encoder, tokenizer):
    with torch.no_grad():
        text_inputs = tokenizer(
            prompt_batch,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )
    # print('qwq1', prompt_embeds[0].shape)
    # print('qwq2', prompt_embeds[1].shape)
    return prompt_embeds[0]


pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'

tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer",
    revision=None, cache_dir='cache'
)
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder",
    revision=None, cache_dir='cache'
)

text_encoder.to('cuda')


def wapper_encode_prompt(text):
    return encode_prompt(text, text_encoder, tokenizer)


_temp_test_ret = wapper_encode_prompt(('qaq', 'qwq', 'qaq is good.', 'qwq'))

data_info_src = os.path.join(raw_msrvtt_src, 'train_val_videodatainfo.json')
with open(data_info_src, 'r') as f:
    data_info = json.loads(f.read())

dataset_src = [(video['video_id']) for video in data_info['videos']]

# prompts[video_id] is the list of the prompts of video_id
prompts = {}
for sentence in data_info['sentences']:
    if prompts.get(sentence["video_id"]):
        prompts[sentence["video_id"]].append(sentence["caption"])
    else:
        prompts[sentence["video_id"]] = [sentence["caption"]]

sink = wds.ShardWriter("dataset/data/msrvtt-webdataset.shard-%06d.tar", maxcount=500, maxsize=1e8)

for video_id in tqdm(dataset_src):
    
    data_path = os.path.join(raw_msrvtt_src, os.path.join(video_subfolder, video_id + '.mp4'))
    #dataset_src.set_description(f"Processing {video_id}")
    video = cv2.VideoCapture(data_path)

    frames = []
    depth_maps = []
    while True:
        res, frame = video.read()
        if not res:
            break
        # print(frame.shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        # frame -> depth map
        #depth_map pretrained model(frame)
        # depth_maps = prepare_depth_maps(frames)
    text = prompts[video_id]

    # have to use ternsor to represent the text, which is only allowed by accelerator.
    text_emb = wapper_encode_prompt(text)
    
    if len(frames) < 2:
        print(f"Warning: Video {video_id} has less than 2 frames. Skipping.")
        continue
    else:
        i = random.randint(0, len(frames)-3)
        j = i + 2
    depth_maps = prepare_depth_maps(frames[i], frames[j])
    # print(text)
    # i = 0
    # j = 1
    # sink.write({
    #     "__key__": video_id,
    #     "frames.pyd": [frames[i], frames[j]],
    #     "text_emb.pyd": text_emb,
    #     "text_str.pyd": text,
    #     "index.pyd": [i, j],
    #     "length.cls": len(frames),
    #     "depth_map.pyd": depth_maps
    # })

    if len(frames) > max(i, j):
        sink.write({
            "__key__": video_id,
            "frames.pyd": [frames[i], frames[j]],
            "text_emb.pyd": text_emb,
            "text_str.pyd": text,
            "index.pyd": [i, j],
            "length.cls": len(frames),
            "depth_map.pyd": [depth_maps[0], depth_maps[1]]
        })
    else:
        print(f"Video {video_id} has insufficient frames: {len(frames)} frames available.")
sink.close()