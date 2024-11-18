import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torchvision import models
from typing import Any, Dict, Optional
import segmentation_models_pytorch as smp

from .unet import UNetResNet34 
from .SAM2UNet import get_sam2unet
from .smp_utils import get_smp_model

from ..utils.loss import *

def get_criterion(criterion_name: str) -> nn.Module:
    criterions = {
        'CrossEntropy': cross_entropy_loss,
        'bce': bce_loss,
        'bce+dice': calc_loss_bce_dice,
        'dice': dice_loss,
        'StructureLoss': multiscale_structure_loss,
        'focal+dice': focal_dice_loss,
        'unet3p': unet3p_loss
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

    torchvision_models = {
        "fcn_50": lambda: models.segmentation.fcn_resnet50(**model_config['config']),
        "fcn_101": lambda: models.segmentation.fcn_resnet101(**model_config['config']),
        "deeplabv3_50": lambda: models.segmentation.deeplabv3_resnet50(**model_config['config']),
        "deeplabv3_101": lambda: models.segmentation.deeplabv3_resnet101(**model_config['config']),
    }

    if model_name in torchvision_models:
        model = torchvision_models[model_name]()
        last_channels = 512 if "fcn" in model_name else 256
        model.classifier[4] = nn.Conv2d(last_channels, num_classes, kernel_size=1)
        
    elif 'smp_' in model_name:
        model = get_smp_model(model_config, num_classes)

    elif model_name == "myUnet":
        model = UNetResNet34()
    
    elif 'sam2unet_' in model_name:
        model = get_sam2unet(model_name)  
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model