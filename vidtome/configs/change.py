import yaml
import os
import glob

# 读取YAML文件
def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# 写入YAML文件
def write_yaml(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(data, file, allow_unicode=True)

# 修改YAML文件内容
def modify_yaml_content(yaml_content):
    example = read_yaml("/root/autodl-tmp/research/downstream_task/VidToMe-main/configs/config_example/black-swan.yaml")
    #yaml_content['sd_version'] = '1.5'
    example['input_path'] = yaml_content['input_data']['video_path']
    example['work_dir'] = yaml_content['output_dir']
    example['generation']['prompt'] = yaml_content['validation_data']['prompts']
    example['inversion']['prompt'] = yaml_content['input_data']['prompt']
    return example

def main(input_yaml_path):
    # 原始YAML文件路径
    change_content = read_yaml(input_yaml_path)
    # 修改内容
    modified_yaml_content = modify_yaml_content(change_content)
    # 保存到新的YAML文件
    write_yaml(modified_yaml_content, input_yaml_path)
    print(f"YAML文件已更新并保存到：{input_yaml_path}")

if __name__ == "__main__":
    yaml_files = glob.glob(f'/root/autodl-tmp/research/downstream_task/VidToMe-main/configs/newconfig/*.yaml')
    
    for file_name in yaml_files:
        main(file_name)
