import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

import os
import pandas as pd
import numpy as np
from src.models.model_utils import get_model
from src.datasets.dataloader import get_inference_loaders
from src.utils.rle_convert import encode_mask_to_rle

def run(config):
    method = config['ensemble']['method']
    model_folders = config['ensemble']['models_folders']
    classes = config['classes']
    threshold = config['train']['threshold']
    output_csv_path = os.path.join(config['paths']['output_dir'], f'ensemble_{method}_output.csv')
    device = torch.device(config['device'])

    test_loader = get_inference_loaders(config)
    models = []

    for folder in model_folders:
        for file in os.listdir(folder):
            if file.endswith(".pth") and "_best_" in file:
            
                model_name = file.split("_best_")[0]
                model_path = os.path.join(folder, file)
                model = get_model(model_name, classes).to(device)
                
                pth_ = torch.load(model_path, map_location=device)
                pth_ = pth_['model_state_dict'] if isinstance(pth_, dict) and 'model_state_dict' in pth_ else pth_
                
                model.load_state_dict(pth_)
                model.eval()
                models.append(model)
                print(f"Loaded model: {model_name} from {model_path}")

    CLASS2IND = {v: i for i, v in enumerate(classes)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}
    
    rles = []
    filename_and_class = []
    
    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.to(device)

            outputs_list = []
            for model in models:
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs, *_ = outputs
                outputs = outputs['out'] if isinstance(outputs, dict) and 'out' in outputs else outputs
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                outputs_list.append(outputs)
            
            stacked_outputs = torch.stack(outputs_list)

            
            if method == "soft":
                ensembled = torch.mean(stacked_outputs, dim=0)
            else:  
                binary_outputs = (stacked_outputs > threshold).float()
                ensembled = torch.mean(binary_outputs, dim=0) > 0.5
            
            final_outputs = (ensembled > threshold).cpu().numpy()

            for output, image_name in zip(final_outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    classes, filenames = zip(*[x.split("_") for x in filename_and_class])
    image_names = [os.path.basename(f) for f in filenames]
    df = pd.DataFrame({
        "image_name": image_names,
        "class": classes,
        "rle": rles,
    })
    
    df.to_csv(output_csv_path, index=False)