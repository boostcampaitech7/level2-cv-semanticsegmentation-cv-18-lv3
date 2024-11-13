import timm
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torchvision import models
from typing import Any, Dict, Optional
import segmentation_models_pytorch as smp
import torch.nn.functional as F


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
    

def get_criterion(criterion_name: str) -> nn.Module:
    criterions = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(),
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
        'bce+dice': calc_loss_bce_dice
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
    elif 'smp_' in model_name:
        model_name = model_name.split('_')[1]
        
        if model_name == 'unet':
            model = smp.Unet('resnet34', encoder_weights="imagenet", classes=num_classes)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
    else:
        raise ValueError(f"Unkown model: {model_name}")

    # 매핑된 모델 이름 가져오기, 없으면 원래 이름 사용
    # model_name = model_mapping.get(model_config_name, model_config_name)

    return model