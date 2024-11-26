import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Any, Dict, Tuple

from src.models.CLIPSeg import prepare_conditional

# save model function 
def save_model(state: Dict[str, Any], save_dir: str, file_name: str = "best_model.pth"):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    torch.save(state, output_path)
    print(f"Model saved to {output_path}")
    
def train_one_epoch(
                model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                model_name: str,
                classes: list,
            ) -> Tuple[float, float]:
    
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        inputs, masks = batch
        inputs[0], labels = inputs[0].to(device), masks[0].to(device)
        batch_size, num_classes, height, width = labels.size()
        # print(inputs.shape)
        # print(masks.shape)
        # print(labels.size()) # torch.Size([4, 29, 352, 352])

        # 각 클래스에 대해 반복
        for class_idx in range(num_classes):
            optimizer.zero_grad()
            # 현재 클래스의 바이너리 마스크 추출
            binary_labels = labels[:, class_idx, :, :].unsqueeze(1) # [batch_size, 1, height, width]

            if model_name == 'clipseg':
                # 현재 클래스 이름을 배치 크기만큼 복제하여 리스트로 생성
                phrases = [classes[class_idx]] * batch_size
                # batch_inputs = (inputs[0], phrases)
                cond = prepare_conditional(phrases)
                outputs, visual_q, _, _ = model(inputs[0], cond, return_features=True)
            else :
                outputs = model(inputs)

            logits = outputs['out'] if isinstance(outputs, dict) and 'out' in outputs else outputs
            if model_name == 'clipseg':
                loss = criterion(logits, binary_labels)

            else:
                loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
    
    if model_name == 'clipseg':
        epoch_loss = total_loss / (len(dataloader) * num_classes)
    else:
        epoch_loss = total_loss / len(dataloader)


    return epoch_loss

def validate(
            model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module,
            device: torch.device,
            metric_fn: Any,
            classes: list,
            model_name: str,
            threshold: float = 0.5,
        ) -> Tuple[float, float]:
    
    model.eval()
    total_loss = 0.0 

    dices = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            inputs, masks = batch
            inputs[0], labels = inputs[0].to(device), masks[0].to(device)
            batch_size, num_classes, height, width = labels.shape


            for class_idx in range(num_classes):
                # 현재 클래스의 바이너리 마스크 추출
                binary_labels = labels[:, class_idx, :, :].unsqueeze(1)

                if model_name == 'clipseg':
                    phrases = [classes[class_idx]] * batch_size
                    # batch_inputs = (inputs[0], phrases)
                    cond = prepare_conditional(phrases)
                    outputs, visual_q, _, _ = model(inputs[0], cond, return_features=True)
                else :
                    outputs = model(inputs)


                if isinstance(outputs, tuple) :
                    logits, logits1, logits2 = outputs
                    use_multiple_outputs = True
                else : 
                    logits = outputs['out'] if isinstance(outputs, dict) and 'out' in outputs else outputs
                    use_multiple_outputs = False

                logits_h, logits_w = logits.size(-2), logits.size(-1)
                labels_h, labels_w = labels.size(-2), labels.size(-1)

                #출력과 레이블의 크기가 다른 경우 출력 텐서를 레이블의 크기로 보간
                if logits_h != labels_h or logits_w != labels_w:
                    logits = F.interpolate(logits, size=(labels_h, labels_w), mode="bilinear", align_corners=False)
                    if use_multiple_outputs:
                        logits1 = F.interpolate(logits1, size=(labels_h, labels_w), mode="bilinear", align_corners=False)
                        logits2 = F.interpolate(logits2, size=(labels_h, labels_w), mode="bilinear", align_corners=False)

                mask_input = binary_labels   # if model_name == 'clipseg' else labels

                if use_multiple_outputs:
                    loss = criterion((logits, logits1, logits2), mask_input)
                else:
                    loss = criterion(logits, mask_input)
            
                total_loss += loss.item()

                probs = torch.sigmoid(logits)
                # threshold 추가해서 기준 치 이상만 label로 분류 
                # outputs = (probs > threshold).detach().cpu()
                # masks = masks.detach().cpu()
                outputs = (probs > threshold)

                if model_name == 'clipseg':
                    dice = metric_fn.calculate(outputs, binary_labels).detach().cpu()
                else:
                    dice = metric_fn.calculate(outputs, labels).detach().cpu()
                # dice = metric_fn.calculate(outputs, binary_labels).detach().cpu()
                dices.append(dice)

    if model_name == 'clipseg':
        epoch_loss = total_loss / (len(dataloader) * num_classes)

        dices = torch.cat(dices, 0)
        avg_dice = torch.mean(dices).item()
    else:
        epoch_loss = total_loss / len(dataloader)
        
        dices = torch.cat(dices, 0)
        dices_per_class = torch.mean(dices, 0)
        dice_str = [
            f"{c:<12}: {d.item():.4f}"
            for c, d in zip(classes, dices_per_class)
        ]
        dice_str = "\n".join(dice_str)
        
        avg_dice = torch.mean(dices_per_class).item()
    
    return epoch_loss, avg_dice

