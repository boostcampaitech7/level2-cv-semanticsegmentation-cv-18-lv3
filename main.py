import yaml
from pathlib import Path
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np

from src import train
from src import inference 


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_config():
    config = {}
    config_files = ['base.yaml', 'train.yaml', 'path.yaml']
    for file in config_files:
        with open(f'configs/{file}', 'r') as f:
            config.update(yaml.safe_load(f))

    if config['training'] == 'cuda' and not torch.cuda.is_available():
        config['training'] = 'cpu'
    
    return config


if __name__ == "__main__":
    config = get_config()


    set_random_seed(config['random_seed'])
    mode = config['mode']

    if mode == 'train':
        train.run(config)
    elif mode == 'inference':
        inference.run(config)

    else:
        raise ValueError(f"Invalid mode: {mode}")