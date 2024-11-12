import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import random
import numpy as np

# save model function 
def save_model(model, save_dir: str, file_name: str = "best_model.pth"):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


def train_one_epoch(model, dataloader, criterion, optimizer, device, metric_fn):
    model.train()
    total_loss = 0
    all_outputs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # valid_mask = labels != -1
        # if not valid_mask.any():
        #     continue  # 유효한 샘플이 없으면 이 배치를 건너뜁니다
            
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

    epoch_loss = total_loss / len(dataloader)

    # 메트릭 계산
    y_true = torch.cat(all_labels, dim=0)
    y_pred = torch.cat(all_outputs, dim=0)

    metric_value = metric_fn.calculate(y_pred, y_true).mean().item()

    return epoch_loss, metric_value

def validate(model, dataloader, criterion, device, metric_fn):
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

            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_outputs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    epoch_loss = total_loss / len(dataloader)

    y_true = torch.cat(all_labels, dim=0)
    y_pred = torch.cat(all_outputs, dim=0)

    metric_value = metric_fn.calculate(y_pred, y_true).mean().item()

    return epoch_loss, metric_value

# def calculate_class_loss_metric(y_true, y_pred, criterion, metric_fn):
#     classes = np.unique(y_true)
#     class_losses = {}
#     class_metric = {}

#     for cls in classes:
#         indices = np.where(y_true == cls)
#         size = len(indices[0])
#         if size == 0:
#             continue

#         class_labels = y_true[indices]
#         class_preds = y_pred[indices]
        
#         class_labels_tensor = torch.tensor(class_labels).to(y_pred.device)
#         class_preds_tensor = y_pred[indices]
#         loss = criterion(class_preds_tensor, class_labels_tensor).item()
#         class_losses[cls] = loss / size

#         metric = metric_fn.calculate(class_preds.cpu().numpy(), class_labels)
#         class_metric[cls] = metric

#     return class_losses, class_metric