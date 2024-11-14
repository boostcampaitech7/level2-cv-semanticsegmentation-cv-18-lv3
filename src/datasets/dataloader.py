import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Any
import albumentations as A
from src.datasets.dataset import XRayDataset
from src.utils.augmentation import get_transform, load_config # get_augmentation 함수 가져오기

def get_data_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:

    train_transforms = get_transform(config['data'], is_train=True)
    val_transforms = get_transform(config['data'], is_train=False)

    train_dataset = XRayDataset(
        image_root=config['paths']['train_image_root'],
        label_root=config['paths']['train_label_root'],
        classes=config['classes'],
        mode='train',
        transforms=train_transforms
    )
    
    val_dataset = XRayDataset(
        image_root=config['paths']['train_image_root'],
        label_root=config['paths']['train_label_root'],
        classes=config['classes'],
        mode='val',
        transforms=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['train']['batch_size'],
        shuffle=True,
        num_workers=config['data']['train']['num_workers'],
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['val']['batch_size'],
        shuffle=False,
        num_workers=config['data']['val']['num_workers'],
        drop_last=False
    )

    return train_loader, val_loader


def get_inference_loaders(config: dict[str, Any]) -> DataLoader:
    
    inference_transforms = get_transform(config, is_train=False)
    
    inference_dataset = XRayDataset(
        image_root=config['paths']['inference_image_root'],
        label_root=config['paths']['inference_label_root'],
        classes=config['classes'],
        mode='inference',
        transforms=inference_transforms

    )
    
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=config['data']['inference']['batch_size'],
        shuffle=False,
        num_workers=config['data']['inference']['num_workers'],
        drop_last=False
    )
    
    return inference_loader
    