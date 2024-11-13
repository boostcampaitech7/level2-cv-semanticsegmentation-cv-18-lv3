import yaml
from pathlib import Path
import torch
import random
import numpy as np
import glob
import argparse
import os
from src import train
from src import inference 
from datetime import datetime
import shutil
from typing import Any, Dict

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_config(config_folder):
    config = {}

    config_folder = os.path.join(config_folder,'*.yaml')
    
    config_files = glob.glob(config_folder)
    
    for file in config_files:
        with open(file, 'r') as f:
            config.update(yaml.safe_load(f))
    
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print('using cpu now...')
        config['device'] = 'cpu'

    return config

def save_config(config: Dict[str, Any], output_dir: str):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    folder_name = f"{timestamp}_{config['model']['name']}_{config['developer']}"
    folder_path = os.path.join(output_dir, folder_name)
    
    os.makedirs(folder_path, exist_ok=True)
    
    output_path = os.path.join(folder_path, 'config.yaml')
    
    with open(output_path, 'w') as file:
        yaml.dump(config, file)
    
    print(f"Config file saved to {output_path}")
    return folder_path

if __name__ == "__main__":
    is_debug = True
    
    if is_debug:
        config_folder = "configs"
        mode = 'train'
    else:
        parser = argparse.ArgumentParser(description='Parse configuration files from a folder')
        parser.add_argument('--mode', required=True, help="Select mode(train/inference/dev)")
        parser.add_argument('--config-folder', required=True, help="Path to config folder containing YAML files")
        args = parser.parse_args()

        config_folder = args.config_folder
        mode = args.mode

    config = get_config(config_folder)

    set_random_seed(config['random_seed'])

    if mode == 'train':
        config['paths']['output_dir'] = save_config(config, config['paths']['output_dir'])
        train.run(config)
        
    elif mode == 'inference':
        inference.run(config)

    else:
        raise ValueError(f"Invalid mode: {mode}")