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
    yaml_content['pretrained_model_path'] = '/root/autodl-tmp/cache_huggingface/huggingface/hub/models--runwayml--stable-diffusion-v1-5/'
    #/root/autodl-tmp/cache_huggingface/huggingface/hub/models--stabilityai--stable-diffusion-2-1-base/
    return yaml_content

def main(input_yaml_path):
    # 原始YAML文件路径
    change_content = read_yaml(input_yaml_path)
    # 修改内容
    modified_yaml_content = modify_yaml_content(change_content)
    # 保存到新的YAML文件
    write_yaml(modified_yaml_content, input_yaml_path)
    print(f"YAML文件已更新并保存到：{input_yaml_path}")

if __name__ == "__main__":
    yaml_files = glob.glob(f'/root/autodl-tmp/research/downstream_task/vid2vid-zero-main/configs/*.yaml')
    
    for file_name in yaml_files:
        main(file_name)