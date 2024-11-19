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
    threshold = config['train']['threshold']
    classes = config['classes']
    CLASS2IND = {v: i for i, v in enumerate(classes)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}
    output_dir = os.path.join(config['paths']['output_dir'],'output.csv')
    device = torch.device(config['device'])

    model = get_model(config['model'], classes).to(device)
    
    test_loader = get_inference_loaders(config)

    model_name = config['model']['name']
    model_path = os.path.join(config['paths']['output_dir'], f"{model_name}_best_model.pth")
    
    pth_ = torch.load(model_path, map_location='cpu')
    pth_ = pth_['model_state_dict'] if isinstance(pth_, dict) and 'model_state_dict' in pth_.keys() else pth_
    
    model.load_state_dict(pth_)
    model.to(device)
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple) :
                outputs, *_ = outputs
            outputs = outputs['out'] if isinstance(outputs, dict) and 'out' in outputs else outputs
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > threshold).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
    
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    df.to_csv(output_dir, index=False)