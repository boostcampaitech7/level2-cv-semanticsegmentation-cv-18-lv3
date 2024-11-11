import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import numpy as np
import datetime
import uuid
import os
from src.utils.trainer import train_one_epoch, validate
from src.utils.params import get_params
from src.utils.data_loaders import get_data_loaders, get_test_loaders
from src.utils.metrics import get_metric_function
from src.models.model_utils import *

def get_wandb_config(config):
    # wandb config 값 가져오기
    wandb_config = {
        "model_name": config['model']['name'],
        "batch_size": config['training']['batch_size'],
        "learning_rate": config['training']['learning_rate'],
        "optimizer": config['training']['optimizer'],
        "weight_decay": config['training']['weight_decay'],
        "lr_scheduler": config['training']['lr_scheduler']['name'],
    }
    return wandb_config

def run(config, trial_number=None):
    os.makedirs(config['paths']['save_dir'], exist_ok=True)
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    model_name = config['model']['name']
    user_name = config['wandb']['user_name']
    team_name = config['wandb']['team_name']

    project_name = f"{model_name}_{user_name}_{current_date}"

    wandb_config = get_wandb_config(config)

    # wandb 초기화
    wandb.init(project=project_name, entity=team_name, config=wandb_config)

    device = torch.device(config['device'])
    model = get_model(config).to(device)

    train_loader, val_loader = get_data_loaders(config, batch_size=config['training']['batch_size'])

    criterion = get_criterion(config['training']['criterion'])
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_lr_scheduler(optimizer, config['training']['lr_scheduler'])

    metric_fn = get_metric_function(config['training']['metric'])

    best_val_metric = metric_fn.worst_value
    patience_counter = 0
    early_stopping_config = config['training']['early_stopping']

    #메인 트레이닝
    for epoch in range(config['training']['num_epochs']):
        train_loss, train_metric, train_class_losses, train_class_metric = train_one_epoch(model, train_loader, criterion, optimizer, device, metric_fn)
        val_loss, val_metric, val_class_losses, val_class_metric = validate(model, val_loader, criterion, device, metric_fn)

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")

        # wandb log 항목별 작성
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_metric": train_metric,
            "val_loss": val_loss,
            "val_metric": val_metric,
            # "train_class_metric": train_class_metric,
            # "val_class_metric": val_class_metric
        })

        #print("Train Class Losses:", train_class_losses)
        #print("Train Class metric:", train_class_metric)
        #print("Val Class Losses:", val_class_losses)
        #print("Val Class metric:", val_class_metric)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if config['training']['lr_scheduler']['monitor'] == 'loss':
                scheduler.step(val_loss)
            else:
                scheduler.step(val_metric)
        else:
            scheduler.step()

        early_stop_value = val_loss if config['training']['early_stopping']['monitor'] == 'loss' else val_metric
        if metric_fn.is_better(early_stop_value, best_val_metric, early_stopping_config['min_delta']):
            best_val_metric = early_stop_value
            patience_counter = 0
            if trial_number is not None:
                model_path = f"{config['paths']['save_dir']}/{model_name}_best_model_trial_{trial_number + 1}.pth"
            else:
                model_path = f"{config['paths']['save_dir']}/{model_name}_best_model.pth"
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_config['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    
    
    wandb.finish()
    return best_val_metric