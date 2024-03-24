import os
import json
from openai import OpenAI
from random import choice
from tqdm import tqdm 

client = OpenAI(api_key="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")
directory = 'data\set'

def modify_caption(caption):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Modify this caption by reasonably changing nouns, adding words, or adding imaginative style: [{caption}]"}
    ]
    )
    return response['choices'][0]['message']['content']


for filename in tqdm(os.listdir(directory), desc='Processing JSON files'):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        data['caption_0'] = modify_caption(data['caption_0'])

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

print("所有caption_0已经被修改。")
