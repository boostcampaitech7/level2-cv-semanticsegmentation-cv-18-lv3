import os
import subprocess
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torchvision import models
from typing import Any, Dict, Optional
import segmentation_models_pytorch as smp

from .unet import UNetResNet34 
from .SAM2UNet import SAM2UNet

from ..utils.loss import *

def get_criterion(criterion_name: str) -> nn.Module:
    criterions = {
        'CrossEntropy': cross_entropy_loss,
        'bce': bce_loss,
        'bce+dice': calc_loss_bce_dice,
        'dice' : dice_loss,
        'StructureLoss' : multiscale_structure_loss,
        'focal+dice' : focal_dice_loss 
    }
    if criterion_name in criterions:
        return criterions[criterion_name]
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")

def get_optimizer(optimizer_config : Dict[str, Any], parameters) -> optim.Optimizer :
    optimizer_name = optimizer_config['name']

    if optimizer_name == 'Adam':
        # lr, weight_decay
        optimizer = optim.Adam(parameters, **optimizer_config['config'])
    elif optimizer_name == 'SGD':
        # lr, momentum, weight_decay
        optimizer = optim.SGD(parameters, **optimizer_config['config'])
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(parameters, **optimizer_config['config'])
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer

def get_lr_scheduler(optimizer: optim.Optimizer, scheduler_config: Dict[str, Any]):
    scheduler_name = scheduler_config['name']
    monitor = scheduler_config.get('monitor', 'metric')

    if scheduler_name == 'ReduceLROnPlateau':
        # factor, patience, min_lr, monitor

        mode = 'min' if monitor == 'loss' else 'max'

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            **scheduler_config['config']
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler

# timm 라이브러리에서 모델 불러오기
def get_model(model_config: Dict[str, Any], classes) -> nn.Module:
    model_name = model_config['name']
    num_classes = len(classes)

    if model_name == 'fcn_50':
        model = models.segmentation.fcn_resnet50(**model_config['config'])
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    elif model_name == 'fcn_101':
        model = models.segmentation.fcn_resnet101(**model_config['config'])
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    elif model_name == 'deeplabv3_50':
        model = models.segmentation.deeplabv3_resnet50(**model_config['config'])
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)        
    elif model_name == 'deeplabv3_101':
        model = models.segmentation.deeplabv3_resnet101(**model_config['config'])
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    elif model_name == "myUnet":
        model = UNetResNet34()
        
    elif 'smp_' in model_name:
        model_name = model_name.split('_')[1]
        if model_name == 'unet':
            model = smp.Unet('resnet34', encoder_weights="imagenet", classes=num_classes)
        elif model_name == 'maxvit':
            model = smp.Unet("tu-maxvit_tiny_tf_512",  encoder_weights="imagenet", in_channels=3, classes=29)
        elif model_name == 'unet++':
            model = smp.UnetPlusPlus('resnet152', encoder_weights="imagenet", classes=num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    elif model_name == "myUnet":
        model = UNetResNet34()

    elif 'sam2unet_' in model_name :
        model_size = model_name.split('_')[1]
        hiera_dir = './pretrained_models'
        if model_size == 'tiny' :
            hiera_file = 'sam2_hiera_tiny.pt'
            download_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt'
        elif model_size == 'base' :
            hiera_file = 'sam2_hiera_base+.pt'
            download_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt'
        elif model_size == 'large' :
            hiera_file = 'sam2_hiera_large.pt'
            download_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
        else :
            raise ValueError(f"Unknown model: {model_name}")
        
        os.makedirs(hiera_dir, exist_ok=True)
        hiera_path = os.path.join(hiera_dir, hiera_file)

        if not os.path.exists(hiera_path) :
            print(f"Hiera file not found at {hiera_path}. Downloading from {download_url}...")
            try:
                subprocess.run(['wget', '-O', hiera_path, download_url], check=True)
            except subprocess.CalledProcessError as e :
                raise RuntimeError(f"Failed to download the hiera file from {download_url}. Error: {e}")

        model = SAM2UNet(model_size, hiera_path)
    else:
        raise ValueError(f"Unkown model: {model_name}")

    # 매핑된 모델 이름 가져오기, 없으면 원래 이름 사용
    # model_name = model_mapping.get(model_config_name, model_config_name)

    return model