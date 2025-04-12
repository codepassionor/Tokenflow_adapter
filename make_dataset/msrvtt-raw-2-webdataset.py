import webdataset as wds
import cv2
import os
import json
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

raw_msrvtt_src = '/data/MSRVTT'
sink = wds.TarWriter("/data/msrvtt-webdataset.tar")
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
    noise0 = torch.randn(3, height, width)
    noise1 = torch.randn(3, height, width)
    noise = torch.cat((noise0, noise1, noise0, noise1), dim=2)

    if(video_id[-1] == '0'): print(noise.shape)

    text = prompts[video_id]

    # have to use ternsor to represent the text, which is only allowed by accelerator.
    text = wapper_encode_prompt(text)

    sink.write({
        "__key__": video_id,
        "input.ppm": final_image,
        "text.pyd": text,
        "noise.pyd": noise,
    })

sink.close()



