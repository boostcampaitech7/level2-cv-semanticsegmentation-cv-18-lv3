import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, Any, Tuple
from utils.datasets import XRayDataset, get_transform

def get_data_loaders(config: Dict[str, Any], batch_size: int = 8) -> Tuple[DataLoader, DataLoader]:
    if batch_size is None:
        batch_size = config['training']['batch_size']
    
    train_info_file, val_info_file = split_and_save_dataset(config)
    
    train_dataset = XRayDataset(
        is_train = True,
        transforms = tf
    )
    
    val_dataset = XRayDataset(
        is_train = False,
        transforms = tf
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

 

    return train_loader, val_loader