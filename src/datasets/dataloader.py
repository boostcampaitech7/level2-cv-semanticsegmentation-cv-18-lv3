import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, Any, Tuple
from utils.datasets import XRayDataset, get_transform

def get_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    batch_size = config['train']['batch_size']
 
    tf = A.Resize(512, 512) #추후 Augmentation 영역으로 빼야함

    train_dataset = XRayDataset(
        image_root=config['paths']['train_image_root']
        label_root=config['paths']['train_label_root']
        classes=config['classes']
        mode='train'
        transforms=tf
    )
    
    val_dataset = XRayDataset(
        image_root=config['paths']['train_image_root']
        label_root=config['paths']['train_label_root']
        classes=config['classes']
        mode='val'
        transforms=tf
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

    return train_loader, val_loader


def get_inference_loaders(config: Dict[str, Any]) -> DataLoader:
    
    tf = A.Resize(512, 512)
    
    inference_dataset = XRayDataset(
        image_root=config['paths']['test_image_root']
        label_root=config['paths']['test_label_root']
        classes=config['classes']
        mode='test'
        transforms=tf
    )
    
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    return inference_loader
    