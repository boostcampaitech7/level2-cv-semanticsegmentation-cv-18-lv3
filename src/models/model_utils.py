import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torchvision import models
from typing import Any, Dict, Optional
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from .unet import UNetResNet34 
from .SAM2UNet import get_sam2unet
from .smp_utils import get_smp_model

def calc_loss_bce_dice(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def focal_loss(inputs, targets, alpha=.25, gamma=2) : 
    BCE = F.binary_cross_entropy_with_logits(inputs, targets)
    BCE_EXP = torch.exp(-BCE)
    loss = alpha * (1-BCE_EXP)**gamma * BCE
    return loss

def focal_dice_loss(pred=None, target=None, focal_weight = 0.5):
    focal = focal_loss(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = focal * focal_weight + dice * (1 - focal_weight)
    return loss

    

def structure_loss(pred, target):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
    wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * target) * weit).sum(dim=(2, 3))
    union = ((pred + target) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def multiscale_structure_loss(pred, target):
    loss = 0
    pred0, pred1, pred2 = pred
    loss0 = structure_loss(pred0, target)
    loss1 = structure_loss(pred1, target)
    loss2 = structure_loss(pred2, target)
    loss = loss0 + loss1 + loss2
    return loss

def get_criterion(criterion_name: str) -> nn.Module:
    criterions = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(),
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
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