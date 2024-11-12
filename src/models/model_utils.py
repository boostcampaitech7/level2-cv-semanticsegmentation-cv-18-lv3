import timm
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torchvision import models
from typing import Any, Dict, Optional

def get_criterion(criterion_name: str) -> nn.Module:
    criterions = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(),
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
    }
    if criterion_name in criterions:
        return criterions[criterion_name]
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")


def get_optimizer(config : Dict[str, Any], parameters) -> optim.Optimizer :
    optimizer_config = config['train']['optimizer']
    optimizer_name = optimizer_config['name']
    lr = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', 0.0)
    momentum = optimizer_config.get('momentum', 0.0)

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer

def get_lr_scheduler(optimizer: optim.Optimizer, scheduler_config: Dict[str, Any]):
    scheduler_name = scheduler_config['name']

    if scheduler_name == 'ReduceLROnPlateau':
        factor = scheduler_config.get('factor', 0.1)
        patience = scheduler_config.get('patience', 3)
        min_lr = scheduler_config.get('min_lr', 1e-6)
        monitor = scheduler_config.get('monitor', 'metric')

        mode = 'min' if monitor == 'loss' else 'max'

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler

# timm 라이브러리에서 모델 불러오기
def get_model(config):
    model_name = config['model']['name']
    num_classes = len(config['classes'])

    if model_name == 'fcn_50':
        model = models.segmentation.fcn_resnet50(pretrained=config['model']['pretrained'])
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    # 매핑된 모델 이름 가져오기, 없으면 원래 이름 사용
    #model_name = model_mapping.get(model_config_name, model_config_name)


    return model
