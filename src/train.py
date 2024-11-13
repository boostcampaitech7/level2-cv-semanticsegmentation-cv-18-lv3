import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import datetime
import uuid
import os
from src.utils.trainer import train_one_epoch, validate, save_model
from src.datasets.dataloader import get_data_loaders
from src.utils.metrics import get_metric_function
from src.models.model_utils import *
from src.utils.wandb_logger import init_wandb, log_metrics, finish_wandb
from typing import Any, Dict


def run(config: Dict[str, Any]) -> float:
    model_name = config['model']['name']
    threshold = config['train']['threshold']

    init_wandb(config)

    device = torch.device(config['device'])
    model = get_model(config['model'], config['classes']).to(device)

    train_loader, val_loader = get_data_loaders(config)

    criterion = get_criterion(config['train']['criterion']['name'])
    optimizer = get_optimizer(config['train']['optimizer'], model.parameters())
    scheduler = get_lr_scheduler(optimizer, config['train']['lr_scheduler'])
    metric_fn = get_metric_function(config['train']['metric']['name'])

    best_val_metric = metric_fn.worst_value
    patience_counter = 0
    early_stopping_config = config['train']['early_stopping']
    
    val_step = config['data']['val']['val_step']
    val_loss, val_metric = float('INF'), 0
    #메인 트레이닝
    for epoch in range(config['train']['num_epochs']):
        train_loss= train_one_epoch(model, train_loader, criterion, optimizer, device)

        print(f"Epoch {epoch+1}/{config['train']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}")
        
        if (epoch+1) % val_step == 0:
            val_loss, val_metric = validate(model, val_loader, criterion, device, metric_fn, config['classes'] ,threshold)
            print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")
            
            #val_step 마다 wandb에 기록 
            log_metrics(epoch, train_loss, val_loss, val_metric)
        else:
            #val_step 이 아닌 경우에는 train_loss 만 기록
            log_metrics(epoch, train_loss) 

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

            save_model(model, config['paths']['output_dir'], f"{model_name}_best_model.pth")
        else:
            patience_counter += ((epoch+1) % val_step == 0)

        if patience_counter >= early_stopping_config['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    finish_wandb()
    return best_val_metric