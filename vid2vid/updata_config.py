import yaml
import os
import re
def read_markdown_data_pairs(md_file_path):
    with open(md_file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
        data_pairs = re.findall(r'^(.+)\n(.+)$', md_content, re.MULTILINE)
    return data_pairs

with open('./configs/example.yaml', 'r') as file:
    config = yaml.safe_load(file)

md_file_path = 'data/data_new/prompt.md'
data_pairs = read_markdown_data_pairs(md_file_path)

for idx, (input_prompt, validation_prompt) in enumerate(data_pairs):
    config['output_dir'] = f"outputs/video{idx:02d}"
    config['input_data']['video_path'] = f"data/data_new/video{idx}.mp4"
    config['input_data']['prompt'] = input_prompt
    config['validation_data']['prompts'][0] = validation_prompt

    new_yaml_file = f'./configs/config_new_{idx:02d}.yaml'
    with open(new_yaml_file, 'w') as file:
        yaml.safe_dump(config, file)

    print(f'Created {new_yaml_file}')
