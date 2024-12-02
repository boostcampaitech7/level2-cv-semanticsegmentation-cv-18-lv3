import yaml
from pathlib import Path
import torch
import random
import numpy as np
import glob
import argparse
import os
from src import train, inference, ensemble
from datetime import datetime
import shutil
from typing import Any, Dict

from etc.dev.dev_utils import dev_paths_setting, dev_wandb_setting

def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_config(config_folder: str) -> dict[str, Any]:
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

if __name__ == "__main__":
    is_debug = False
    
    if is_debug:
        config_folder = "outputs/dev_smp_unet_kh"
        mode = 'train'
        dev = True
        resume = False
        pth_path = None
    else:
        parser = argparse.ArgumentParser(description='Parse configuration files from a folder')
        parser.add_argument('-m', '--mode', help="Select mode(train/inference/ensemble)", default="train")
        parser.add_argument('-cf', '--config-folder', help="Path to config folder containing YAML files", default="./configs/")
        parser.add_argument('-d', '--dev', help="dev mode on off", action='store_true', )
        parser.add_argument('-r', '--resume', help="resume train", action='store_true')
        parser.add_argument('-p', '--pth_path', help="path to pth file")
        parser.add_argument('-wh', '--webhook', help='slack webhook alarm', default=False)
        args = parser.parse_args()
        
        config_folder = args.config_folder
        mode = args.mode
        dev = args.dev
        resume = args.resume
        pth_path = args.pth_path
        webhook = args.webhook
   
    config = get_config(config_folder)

    set_random_seed(config['random_seed'])
    
    # dev 환경설정
    if dev:
        dev_paths_setting(config['paths'])
        dev_wandb_setting(config['wandb'])

    if mode == 'train':
        # save_config(config, config['paths']['output_dir'], dev)
        train.run(config, resume, pth_path, dev, webhook)
    elif mode == 'inference':
        inference.run(config)
    elif mode == 'ensemble':
        ensemble.run(config)
    else:
        raise ValueError(f"Invalid mode: {mode}")