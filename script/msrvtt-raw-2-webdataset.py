import webdataset as wds
import cv2
import os
import json
from PIL import Image
import torch
from tqdm import tqdm

raw_msrvtt_src = '/root/autodl-tmp/MSRVTT'
sink = wds.TarWriter("msrvtt.tar"),
video_subfolder = 'video'

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

for video_id in total_dataset_src:
    data_path = os.path.join(raw_msrvtt_src, os.path.join(video_subfolder, video_id + '.mp4'))
    total_dataset_src.set_description(f"Processing {video_id}")
    video = cv2.VideoCapture(data_path)
    success0, frame0 = video.read()
    success1, frame1 = video.read()
    video.release()

    assert success0 and success1

    frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    pil_image0 = Image.fromarray(frame0_rgb)
    pil_image1 = Image.fromarray(frame1_rgb)

    width, height = pil_image0.size

    final_image = Image.new("RGB", (width * 4, height))

    final_image.paste(pil_image0, (width * 0, 0))
    final_image.paste(pil_image1, (width * 1, 0))
    final_image.paste(pil_image0, (width * 2, 0))
    final_image.paste(pil_image1, (width * 3, 0))


    torch.manual_seed(hash(video_id))
    noise = torch.randn(3, height, width)

    text = prompts[video_id]

    sink.write({
        "__key__": video_id,
        "input.ppm": final_image,
        "text.json": text,
        "noise.pyd": noise,
    })

sink.close()



