import argparse
from omegaconf import OmegaConf, DictConfig
import os
import glob

def load_config(i, print_config = True):
    parser = argparse.ArgumentParser()
    yaml_files = glob.glob(f'/root/autodl-tmp/lora/Tokenflow_adapter/vidtome/configs/*.yaml')
    parser.add_argument('--config', type=str,
                        default=yaml_files[i],
                        help="Config file path")
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int, default=300)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    lora_begin = args.begin
    lora_end = args.end
    # Recursively merge base configs
    cur_config_path = args.config
    cur_config = config
    while "base_config" in cur_config and cur_config.base_config != cur_config_path:
        base_config = OmegaConf.load(cur_config.base_config)
        config = OmegaConf.merge(base_config, config)
        cur_config_path = cur_config.base_config
        cur_config = base_config

    prompt = config.generation.prompt
    if isinstance(prompt, str):
        prompt = {"edit": prompt}
    config.generation.prompt = prompt
    OmegaConf.resolve(config)
    if print_config:
        print("[INFO] loaded config:")
        print(OmegaConf.to_yaml(config))
    
    return config, lora_begin, lora_end

def load_config_example(print_config = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default=f'configs/dog.yaml',
                        help="Config file path")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # Recursively merge base configs
    cur_config_path = args.config
    cur_config = config
    while "base_config" in cur_config and cur_config.base_config != cur_config_path:
        base_config = OmegaConf.load(cur_config.base_config)
        config = OmegaConf.merge(base_config, config)
        cur_config_path = cur_config.base_config
        cur_config = base_config

    prompt = config.generation.prompt
    if isinstance(prompt, str):
        prompt = {"edit": prompt}
    config.generation.prompt = prompt
    OmegaConf.resolve(config)
    if print_config:
        print("[INFO] loaded config:")
        print(OmegaConf.to_yaml(config))
    
    return config

def save_config(config: DictConfig, path, gene = False, inv = False):
    os.makedirs(path, exist_ok = True)
    config = OmegaConf.create(config)
    if gene:
        config.pop("inversion")
    if inv:
        config.pop("generation")
    OmegaConf.save(config, os.path.join(path, "config.yaml"))