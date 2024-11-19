import sys
import os
sys.path.append("/data/ephemeral/home/kwak/level2-cv-semanticsegmentation-cv-18-lv3")
from src.models.model_utils import *
from src.datasets.dataset import XRayDataset
from src.utils.augmentation import get_transform
from src.utils.metrics import get_metric_function
from torch.utils.data import Dataset, DataLoader, Subset
from src.utils.rle_convert import encode_mask_to_rle
import pandas as pd
import torch.nn.functional as F
import glob
from tqdm import tqdm
import yaml


def get_config(config_folder):
    config = {}

    config_folder = os.path.join(config_folder,'*.yaml')
    
    config_files = glob.glob(config_folder)
    
    for file in config_files:
        with open(file, 'r') as f:
            config.update(yaml.safe_load(f))
    
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print('using cpu now...')
        config['device'] = 'cpu'

    return config


if __name__ == "__main__":
    img_path = "data/train/DCM"
    label_path = "data/train/outputs_json"
    top_k = 5
    config_path = "outputs/dev_smp_unet_kh"
    pth_path = "outputs/dev_smp_unet_kh/smp_unet_best_model.pth"
    threshold = 0.5
    batch_size = 8
    
    config = get_config(config_path)

    classes = config['classes']
    CLASS2IND = {v: i for i, v in enumerate(classes)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    transform = get_transform(config['data'], is_train=False)

    _dataset = XRayDataset(
        image_root=img_path,
        label_root=label_path,
        classes=config['classes'],
        mode='val',
        transforms=transform
    )

    _dataloader = DataLoader(
        _dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    device = torch.device(config['device'])
    model = get_model(config['model'], config['classes']).to(device)

    checkpoint = torch.load(pth_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = get_criterion(config['train']['criterion']['name'])
    metric_fn = get_metric_function(config['train']['metric']['name'])
    
    dice_list = []
    rles = []
    gt_rles = []
    filename_and_class = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(_dataloader, desc="creating csv")):
            inputs, masks = batch
            inputs, masks = inputs.to(device), masks.to(device)
            
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                    logits, logits1, logits2 = outputs
                    use_multiple_outputs = True
            else: 
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

            probs = torch.sigmoid(logits)
            # threshold 추가해서 기준 치 이상만 label로 분류 
            # outputs = (probs > threshold).detach().cpu()
            # masks = masks.detach().cpu()
            outputs = (probs > threshold)
            dice = metric_fn.calculate(outputs, masks).detach().cpu()
            
            dice_list.append(dice)
            
            # pd에 저장하기 위함
            outputs = outputs.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            for idx, (output, mask) in enumerate(zip(outputs, masks)):
                file_path, _ = _dataset.get_filepath(batch_size*i + idx)
                
                for c, (seg_m, gt_m) in enumerate(zip(output, mask)):
                    rle = encode_mask_to_rle(seg_m)
                    gt_rle = encode_mask_to_rle(gt_m)
                    rles.append(rle)
                    gt_rles.append(gt_rle)
                    filename_and_class.append(f"{IND2CLASS[c]}:{file_path}")

    classes, file_path = zip(*[x.split(":") for x in filename_and_class])
    df_pred = pd.DataFrame({
        "file_path": file_path,
        "class": classes,
        "rle": rles,
        "gt_rle": gt_rles,
    })

    # dice_list = sorted(dice_list, key=lambda x: (-x, x))

    dices = torch.cat(dice_list, 0)

    dices_per_img = torch.mean(dices, 1)

    file_path, _ = _dataset.get_all_path()
    df_dice = pd.DataFrame(dices, columns=[IND2CLASS[i] for i in range(29)])
    df_dice.insert(0, "file_path", file_path)
    
    vis_csv_path = os.path.join(config_path, "vis_csv")
    df_pred_path = os.path.join(vis_csv_path, "pred.csv")
    df_dice_path = os.path.join(vis_csv_path, "dice.csv")
    
    os.makedirs(vis_csv_path, exist_ok=True)
    
    df_pred.to_csv(df_pred_path, index=False)
    df_dice.to_csv(df_dice_path, index=False)