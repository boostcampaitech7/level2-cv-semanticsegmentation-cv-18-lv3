import wandb
import datetime
from typing import Dict, Any
        
class WandbLogger():
    def __init__(self, config: dict[str, any]) -> None:
        if 'wandb' in config.keys(): # wandb_status off : 0, on : 1
            self.init_wandb(config)
            self.wandb_status = True
        else:
            print("wandb not found in config: running train without logger")
            self.wandb_status = False

    @staticmethod
    def init_wandb(config: dict[str, Any]) -> None:
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        model_name = config['model']['name']
        user_name = config['wandb']['user_name']
        team_name = config['wandb']['team_name']
        experiment = config['wandb']['experiment']
        project_name = f"dev_{config['developer']}" if user_name == 'dev' else f"{model_name}"
        
        run_name = f"{model_name}_{user_name}_{experiment}_{current_date}"
        
        # wandb config 
        wandb_config = {
            "model_name": config['model']['name'],
            "batch_size": config['data']['train']['batch_size'],
            "learning_rate": config['train']['optimizer']['config']['lr'],
            "optimizer": config['train']['optimizer']['name'],
            "weight_decay": config['train']['optimizer']['config']['weight_decay'],
            "lr_scheduler": config['train']['lr_scheduler']['name'],
            "learning_rate": config['train']['optimizer']['config']['lr'], 
            "owner" : user_name #누가 돌렸는 지 정보 추가
        }
        
        # wandb initialize 
        try:
            wandb.init(project=project_name, entity=team_name, config=wandb_config,  name=run_name)
        except Exception as e:
            print(f"Error during W&B initialization: {e}")

    def log_metrics(self, epoch: int, train_loss: float, learning_rate : float = None, val_loss: float = None, val_metric: float = None) -> None:
        if self.wandb_status:
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate" : learning_rate
            }
            if val_loss is not None:
                metrics["val_loss"] = val_loss
            if val_metric is not None:
                metrics["val_metric"] = val_metric
            
            try:
                wandb.log(metrics)
            except Exception as e:
                print(f"Error recording W&B logs: {e}")
        
    def finish_wandb(self) -> None:
        if self.wandb_status:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Error during wandb finish: {e}")