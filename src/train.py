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
from src.utils.logger import WandbLogger, save_config
from typing import Any, Dict


def run(config: Dict[str, Any], resume: bool, pth_path: str, dev: bool) -> float:
    model_name = config['model']['name']
    threshold = config['train']['threshold']

    wandb = WandbLogger(config, resume)

    if not(resume):
        save_config(config, "./outputs", dev)
    
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
    
    val_loss, val_metric = float('INF'), 0

    #체크포인트 resume
    start_epoch = 0
    if resume:
        print(f"checkpoint resume ... ")
        checkpoint = torch.load(pth_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 이어서 학습할 epoch 설정
        best_val_metric = checkpoint.get('best_val_metric', best_val_metric)
        print(f"Resuming training from epoch {start_epoch}")

    #메인 트레이닝
    for epoch in range(start_epoch, config['train']['num_epochs']):
        # train
        train_loss= train_one_epoch(model, train_loader, criterion, optimizer, device)
        # validation
        val_loss, val_metric = validate(model, val_loader, criterion, device, metric_fn, config['classes'] ,threshold)
        
        # 현재 학습률 가져오기 
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{config['train']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")
        
        # wandb 기록
        wandb.log_metrics(epoch, train_loss, current_lr, val_loss, val_metric)

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
            
            save_model({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_metric': best_val_metric
            }, config['paths']['output_dir'], f"{model_name}_best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_config['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    wandb.finish_wandb()
    return best_val_metric