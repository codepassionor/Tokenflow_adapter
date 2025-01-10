# Most code is from https://huggingface.co/spaces/Tune-A-Video-library/Tune-A-Video-Training-UI

#!/usr/bin/env python

from __future__ import annotations

import os
from subprocess import getoutput

import gradio as gr
import torch

from gradio_demo.app_running import create_demo
from gradio_demo.runner import Runner

TITLE = '# [vid2vid-zero](https://github.com/baaivision/vid2vid-zero)'

ORIGINAL_SPACE_ID = 'BAAI/vid2vid-zero'
SPACE_ID = os.getenv('SPACE_ID', ORIGINAL_SPACE_ID)
GPU_DATA = getoutput('nvidia-smi')

if os.getenv('SYSTEM') == 'spaces' and SPACE_ID != ORIGINAL_SPACE_ID:
    SETTINGS = f'<a href="https://huggingface.co/spaces/{SPACE_ID}/settings">Settings</a>'
else:
    SETTINGS = 'Settings'

CUDA_NOT_AVAILABLE_WARNING = f'''## Attention - Running on CPU.
<center>
You can assign a GPU in the {SETTINGS} tab if you are running this on HF Spaces.
You can use "T4 small/medium" to run this demo.
</center>
'''

HF_TOKEN_NOT_SPECIFIED_WARNING = f'''The environment variable `HF_TOKEN` is not specified. Feel free to specify your Hugging Face token with write permission if you don't want to manually provide it for every run.
<center>
You can check and create your Hugging Face tokens <a href="https://huggingface.co/settings/tokens" target="_blank">here</a>.
You can specify environment variables in the "Repository secrets" section of the {SETTINGS} tab.
</center>
'''

HF_TOKEN = os.getenv('HF_TOKEN')


def show_warning(warning_text: str) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Box():
            gr.Markdown(warning_text)
    return demo


pipe = None
runner = Runner(HF_TOKEN)

with gr.Blocks(css='gradio_demo/style.css') as demo:
    if not torch.cuda.is_available():
        show_warning(CUDA_NOT_AVAILABLE_WARNING)

    gr.Markdown(TITLE)
    with gr.Tabs():
        with gr.TabItem('Zero-shot Testing'):
            create_demo(runner, pipe)

    if not HF_TOKEN:
        show_warning(HF_TOKEN_NOT_SPECIFIED_WARNING)

demo.queue(max_size=1).launch(share=False)
