import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Any, Dict, Tuple

# save model function 
def save_model(model, save_dir: str, file_name: str = "best_model.pth"):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

def train_one_epoch(
                model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                metric_fn: Any
            ) -> Tuple[float, float]:
    
    model.train()
    total_loss = 0
    all_outputs = []
    all_labels = []

    for idx, batch in enumerate(dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
         # 모델 출력이 딕셔너리인 경우 처리
        logits = outputs['out'] if isinstance(outputs, dict) and 'out' in outputs else outputs

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Sigmoid를 적용하여 확률로 변환
        probs = torch.sigmoid(logits)
        all_outputs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        print(idx,total_loss)

    epoch_loss = total_loss / len(dataloader)

    # 메트릭 계산
    y_true = torch.cat(all_labels, dim=0)
    y_pred = torch.cat(all_outputs, dim=0)

    metric_value = metric_fn.calculate(y_pred, y_true).mean().item()

    return epoch_loss, metric_value

def validate(
            model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module,
            device: torch.device,
            metric_fn: Any,
            threshold: float = 0.5
        ) -> Tuple[float, float]:
    
    model.eval()
    total_loss = 0.0 
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            logits = outputs['out'] if isinstance(outputs, dict) and 'out' in outputs else outputs

            logits_h, logits_w = logits.size(-2), logits.size(-1)
            labels_h, labels_w = labels.size(-2), labels.size(-1)

            #출력과 레이블의 크기가 다른 경우 출력 텐서를 레이블의 크기로 보간
            if logits_h != labels_h or logits_w != logits_w:
                logits = F.interpolate(logits, size=(labels_h, labels_w), mode="bilinear", align_corners=False)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            # threshold 추가해서 기준 치 이상만 label로 분류 
            preds = (probs > threshold).float() 
            all_outputs.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    epoch_loss = total_loss / len(dataloader)

    y_true = torch.cat(all_labels, dim=0)
    y_pred = torch.cat(all_outputs, dim=0)

    metric_value = metric_fn.calculate(y_pred, y_true).mean().item()

    return epoch_loss, metric_value

