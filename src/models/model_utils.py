import timm
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

def get_criterion(criterion_name):
    if criterion_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

def get_optimizer(config, model_parameters):
    optimizer_name = config['training']['optimizer']
    if optimizer_name == "Adam":
        return optim.Adam(model_parameters, lr=config['training']['learning_rate'], 
                          weight_decay=config['training']['weight_decay'])
    elif optimizer_name == "SGD":
        return optim.SGD(model_parameters, lr=config['training']['learning_rate'], 
                         momentum=0.9, weight_decay=config['training']['weight_decay'])
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
    model_config = config['model']
    num_classes = model_config['num_classes']
    
    model_config_name = model_config['name']

    # 모델 간소화 시 모델 전체 이름 매핑
    model_mapping = {
        "resnet":"resnet50",
        "ViT":"vit_base_patch16_224.augreg_in21k",
        "ConvN":"convnext_base",
        "eff3":"efficientnet_b3",
        "eff4":"efficientnet_b4",
        "eff5":"efficientnet_b5",
        "eff6":"efficientnet_b6",
        "eff7":"efficientnet_b7",
        "densenet": "densenet121"
    }

    # 매핑된 모델 이름 가져오기, 없으면 원래 이름 사용
    model_name = model_mapping.get(model_config_name, model_config_name)

    # 모델 생성
    model = timm.create_model(model_name, pretrained=config['model']['pretrained'], num_classes=num_classes)

    return model
