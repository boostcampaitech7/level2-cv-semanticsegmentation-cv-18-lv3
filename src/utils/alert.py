import requests

def send_slack_notification(webhook_url: str, message: str) -> None:
    payload = {"text": message}
    response = requests.post(webhook_url, json=payload)
    
    if response.status_code != 200:
        raise ValueError(f"Slack webhook 요청 실패: {response.status_code}, 응답: {response.text}")
