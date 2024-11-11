import numpy as np
import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device, metric_fn):
    model.train()
    total_loss = 0
    all_outputs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        valid_mask = labels != -1
        if not valid_mask.any():
            continue  # 유효한 샘플이 없으면 이 배치를 건너뜁니다
            
        optimizer.zero_grad()
        outputs = model(inputs)
        
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs  
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_outputs.extend(logits.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(dataloader)
    epoch_metric = metric_fn.calculate(all_outputs, all_labels)

    class_losses, class_metric = calculate_class_loss_metric(
        np.array(all_labels), 
        torch.tensor(np.array(all_outputs)).to(device), 
        criterion,
        metric_fn
    )

    return epoch_loss, epoch_metric, class_losses, class_metric

def validate(model, dataloader, criterion, device, metric_fn):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs  
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_outputs.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(dataloader)
    epoch_metric = metric_fn.calculate(all_outputs, all_labels)

    class_losses, class_metric = calculate_class_loss_metric(
        np.array(all_labels), 
        torch.tensor(np.array(all_outputs)).to(device), 
        criterion,
        metric_fn
    )

    return epoch_loss, epoch_metric, class_losses, class_metric

def calculate_class_loss_metric(y_true, y_pred, criterion, metric_fn):
    classes = np.unique(y_true)
    class_losses = {}
    class_metric = {}

    for cls in classes:
        indices = np.where(y_true == cls)
        size = len(indices[0])
        if size == 0:
            continue

        class_labels = y_true[indices]
        class_preds = y_pred[indices]
        
        class_labels_tensor = torch.tensor(class_labels).to(y_pred.device)
        class_preds_tensor = y_pred[indices]
        loss = criterion(class_preds_tensor, class_labels_tensor).item()
        class_losses[cls] = loss / size

        metric = metric_fn.calculate(class_preds.cpu().numpy(), class_labels)
        class_metric[cls] = metric

    return class_losses, class_metric