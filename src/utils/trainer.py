import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Any, Dict, Tuple

from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext


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
                scaler: GradScaler = None,
                use_amp: bool = False,
                config: Dict[str, Any] = None
            ) -> Tuple[float, float]:
    
    model.train()
    total_loss = 0

    # AMP 및 GradScaler 설정
    autocast_fn = autocast if use_amp else nullcontext

    for batch in tqdm(dataloader, desc="Training"):
        inputs, masks, texts = batch  # clipseg 사용하려면 texts 추가해야 함
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()

        with autocast_fn():
            #  conditional vector 추가
            if config['mix']: 
                pass # mix 기능은 사용하지 않을 예정
            else:
                # no mix
                prompts = model.sample_prompts(texts, prompt_list=('a photo of a {}',))
                # conditional vector 계산
                cond = model.compute_conditional(prompts)
                cond = cond.to(device)


        outputs = model(inputs, cond)[0] # 모델이 튜플을 반환함. 첫번째 요소만 사용
        
         # 모델 출력이 딕셔너리인 경우 처리
        logits = outputs['out'] if isinstance(outputs, dict) and 'out' in outputs else outputs

        loss = criterion(logits, masks)

        # 역전파 및 옵티마이저 업데이트
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()


    epoch_loss = total_loss / len(dataloader)
 
    return epoch_loss

def validate(
            model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module,
            device: torch.device,
            metric_fn: Any,
            classes: list,
            threshold: float = 0.5
        ) -> Tuple[float, float]:
    
    model.eval()
    total_loss = 0.0 

    dices = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            inputs, masks, texts = batch
            inputs, masks = inputs.to(device), masks.to(device)
            # texts는 리스트 형태로 모델에 전달될 프롬프트를 의미

            # 프롬프트 생성
            prompts = model(inputs, prompt_list=('a photo of a {}',))

            # 모델 호출
            outputs = model(inputs, prompts)[0]

            if isinstance(outputs, tuple) :
                logits, logits1, logits2 = outputs
                use_multiple_outputs = True
            else : 
                logits = outputs['out'] if isinstance(outputs, dict) and 'out' in outputs else outputs
                use_multiple_outputs = False

            logits_h, logits_w = logits.size(-2), logits.size(-1)
            labels_h, labels_w = masks.size(-2), masks.size(-1)

            #출력과 레이블의 크기가 다른 경우 출력 텐서를 레이블의 크기로 보간
            if logits_h != labels_h or logits_w != labels_w:
                logits = F.interpolate(logits, size=(labels_h, labels_w), mode="bilinear", align_corners=False)
                if use_multiple_outputs:
                    logits1 = F.interpolate(logits1, size=(labels_h, labels_w), mode="bilinear", align_corners=False)
                    logits2 = F.interpolate(logits2, size=(labels_h, labels_w), mode="bilinear", align_corners=False)

            if use_multiple_outputs:
                loss = criterion((logits, logits1, logits2), masks)
            else:
                loss = criterion(logits, masks)
            
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            # threshold 추가해서 기준 치 이상만 label로 분류 
            # outputs = (probs > threshold).detach().cpu()
            # masks = masks.detach().cpu()
            outputs = (probs > threshold)
            dice = metric_fn.calculate(outputs, masks).detach().cpu()
            dices.append(dice)

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

