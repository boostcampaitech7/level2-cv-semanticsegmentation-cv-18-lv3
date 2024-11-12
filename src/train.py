import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import datetime
import uuid
import os
import random 
from src.utils.trainer import train_one_epoch, validate, save_model
from src.datasets.dataloader import get_data_loaders
from src.utils.metrics import get_metric_function
from src.models.model_utils import *
from src.utils.wandb_logger import init_wandb, log_metrics, finish_wandb
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

def run(config: Dict[str, Any]) -> float:
    
    os.makedirs(config['paths']['save_dir'], exist_ok=True)
    
    model_name = config['model']['name']

    init_wandb(config)

    device = torch.device(config['device'])
    model = get_model(config).to(device)

    train_loader, val_loader = get_data_loaders(config)

    criterion = get_criterion(config['train']['criterion'])
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_lr_scheduler(optimizer, config['train']['lr_scheduler'])

    metric_fn = get_metric_function(config['train']['metric'])

    best_val_metric = metric_fn.worst_value
    patience_counter = 0
    early_stopping_config = config['train']['early_stopping']

    #메인 트레이닝
    for epoch in range(config['train']['num_epochs']):
        train_loss, train_metric = train_one_epoch(model, train_loader, criterion, optimizer, device, metric_fn)
        val_loss, val_metric = validate(model, val_loader, criterion, device, metric_fn)

        print(f"Epoch {epoch+1}/{config['train']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")

        # wandb log 항목별 작성
        log_metrics(epoch, train_loss, train_metric, val_loss, val_metric)


        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if config['train']['lr_scheduler']['monitor'] == 'loss':
                scheduler.step(val_loss)
            else:
                scheduler.step(val_metric)
        else:
            scheduler.step()

        early_stop_value = val_loss if early_stopping_config['monitor'] == 'loss' else val_metric
        if metric_fn.is_better(early_stop_value, best_val_metric, early_stopping_config['min_delta']):
            best_val_metric = early_stop_value
            patience_counter = 0

            save_model(model, config['paths']['save_dir'], f"{model_name}_best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_config['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    finish_wandb()
    return best_val_metric