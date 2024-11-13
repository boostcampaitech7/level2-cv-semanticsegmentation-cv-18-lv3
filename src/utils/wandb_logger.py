import wandb
import datetime
from typing import Dict, Any

def init_wandb(config: Dict[str, Any]) -> None:

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    model_name = config['model']['name']
    user_name = config['wandb']['user_name']
    team_name = config['wandb']['team_name']

    project_name = f"{model_name}_{user_name}_{current_date}"

    run_name = f"{model_name}_{user_name}_{current_date}"
    
    # wandb config 
    wandb_config = {
        "model_name": config['model']['name'],
        "batch_size": config['train']['batch_size'],
        "learning_rate": config['train']['optimizer']['learning_rate'],
        "optimizer": config['train']['optimizer']['name'],
        "weight_decay": config['train']['optimizer']['weight_decay'], 
        "lr_scheduler": config['train']['lr_scheduler']['name'],
    }

    # wandb initialize 
    try:
        wandb.init(project=project_name, entity=team_name, config=wandb_config,  name=run_name)
    except Exception as e:
        print(f"Error during W&B initialization: {e}")

def log_metrics(epoch: int, train_loss: float, val_loss: float = None, val_metric: float = None) -> None:

    metrics = {
        "epoch": epoch,
        "train_loss": train_loss
    }
    if val_loss is not None:
        metrics["val_loss"] = val_loss
    if val_metric is not None:
        metrics["val_metric"] = val_metric
    try:
        wandb.log(metrics)
    except Exception as e:
        print(f"Error recording W&B logs: {e}")

def finish_wandb():
    try:
        wandb.finish()
    except Exception as e:
        print(f"Error during wandb finish: {e}")