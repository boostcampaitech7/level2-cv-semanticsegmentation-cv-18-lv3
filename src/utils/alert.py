import requests
from typing import Dict, Any

def send_slack_notification(config: Dict[str, Any], best_val_metric: float) -> None:
    slack_webhook_url = config['webhook']['url']
    model_name = config['model']['name']
    user_name = config['wandb']['user_name']
    
    message  = f"Training is done!\nBest val metric :{best_val_metric :.4f}"
    slack_message = f"{message}\nModel Name: {model_name}\nUser Name: {user_name}"
    
    payload = {"text": slack_message}
    
    try:
        response = requests.post(slack_webhook_url, json=payload, timeout=10)
        response.raise_for_status()  
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Slack webhook 요청 실패: {e}")
