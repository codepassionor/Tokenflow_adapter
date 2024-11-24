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

raw_msrvtt_src = '/root/autodl-tmp/data/MSRVTT'
video_subfolder = 'video'


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
    revision=None, cache_dir='/root/autodl-tmp/cache'
)
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder",
    revision=None, cache_dir='/root/autodl-tmp/cache'
)
text_encoder.to('cuda')
def wapper_encode_prompt(text):
    return encode_prompt(text, text_encoder, tokenizer)


_temp_test_ret = wapper_encode_prompt(('qaq', 'qwq', 'qaq is good.', 'qwq'))

data_info_src = os.path.join(raw_msrvtt_src, 'MSRVTT_data.json')
with open(data_info_src, 'r') as f:
    data_info = json.loads(f.read())

dataset_src = [ (video['video_id']) for video in data_info['videos'] ]

# prompts[video_id] is the list of the prompts of video_id
prompts = {}
for sentence in data_info['sentences']:
    if prompts.get(sentence["video_id"]):
        prompts[sentence["video_id"]].append(sentence["caption"])
    else:
        prompts[sentence["video_id"]] = [sentence["caption"]]

total_dataset_src = tqdm(dataset_src)



sink = wds.ShardWriter("/root/autodl-tmp/data/msrvtt-webdataset.shard-%06d.tar", maxcount=500, maxsize=1e8)
for video_id in total_dataset_src:
    data_path = os.path.join(raw_msrvtt_src, os.path.join(video_subfolder, video_id + '.mp4'))
    total_dataset_src.set_description(f"Processing {video_id}")
    video = cv2.VideoCapture(data_path)

    frames = []
    depth_maps = []
    while True:
        res, frame = video.read()
        if not res:
            break
        #print(frame.shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        # frame -> depth map
        # depth_map pretrained model(frame)
        depth_maps.append()
        


    text = prompts[video_id]

    # have to use ternsor to represent the text, which is only allowed by accelerator.
    text_emb = wapper_encode_prompt(text)
    #print(text)
    i = 0 
    j = 1
    sink.write({
        "__key__": video_id,
        "frames.pyd": [frames[i], frames[j]],
        "text_emb.pyd": text_emb,
        "text_str.pyd": text,
        "index.pyd": [i, j],
        "length.cls": len(frames)
        # "depth_map.pyd" : depth_map
    })

sink.close()



