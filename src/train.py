import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import datetime
import uuid
import os
from src.utils.trainer import train_one_epoch, validate
from src.datasets.dataloader import get_data_loaders
from src.utils.metrics import get_metric_function
from src.models.model_utils import *
from src.utils.wandb_logger import init_wandb, log_metrics, finish_wandb
from typing import Any, Dict, Optional


def run(config: Dict[str, Any]) -> float:
    os.makedirs(config['paths']['save_dir'], exist_ok=True)
    
    model_name = config['model']['name']

    init_wandb(config)

    device = torch.device(config['device'])
    model = get_model(config).to(device)

    train_loader, val_loader = get_data_loaders(config)

    criterion = get_criterion(config)
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_lr_scheduler(optimizer, config)

    metric_fn = get_metric_function(config['training']['metric'])

    best_val_metric = metric_fn.worst_value
    patience_counter = 0
    early_stopping_config = config['training']['early_stopping']

    #메인 트레이닝
    for epoch in range(config['train']['num_epochs']):
        train_loss, train_metric, train_class_losses, train_class_metric = train_one_epoch(model, train_loader, criterion, optimizer, device, metric_fn)
        val_loss, val_metric, val_class_losses, val_class_metric = validate(model, val_loader, criterion, device, metric_fn)

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

            model_path = f"{config['paths']['save_dir']}/{model_name}_best_model.pth"
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_config['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    finish_wandb()
    return best_val_metric