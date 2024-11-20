import requests
import datetime
from typing import Dict, Any

def send_slack_notification(config: Dict[str, Any], webhook_url: str, message: str) -> None:
  
    model_name = config['model']['name']
    user_name = config['wandb']['user_name']
    
    slack_message = f"{message}\nModel Name: {model_name}\nUser Name: {user_name}"
    
    payload = {"text": slack_message}
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()  # HTTP 에러 발생 시 예외 발생
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Slack webhook 요청 실패: {e}")
