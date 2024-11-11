import timm
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torchvision import models

def get_criterion(criterion_name):
    if criterion_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif criterion_name == "BCE":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

def get_optimizer(config, model_parameters):
    optimizer_name = config['training']['optimizer']
    if optimizer_name == "Adam":
        return optim.Adam(model_parameters, lr=config['optimizer']['learning_rate'], 
                          weight_decay=config['optimizer']['weight_decay'])
    elif optimizer_name == "SGD":
        return optim.SGD(model_parameters, lr=config['optimizer']['learning_rate'], 
                         momentum=config['optimizer']['momentum'], weight_decay=config['optimizer']['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    

def get_lr_scheduler(optimizer, scheduler_config):

    if scheduler_config['name'] == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(scheduler_config['factor']),
            patience=int(scheduler_config['patience']),
            min_lr=float(scheduler_config['min_lr'])
        )

    else:
        raise ValueError(f"Unsupported lr scheduler: {scheduler_config['name']}")

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
