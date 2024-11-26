import os
import json
import random
import datetime
from functools import partial

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from src.utils.augmentation import get_transform

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


class XRayDataset(Dataset):
    def __init__(self, image_root: str, label_root: str, classes: list, mode: str='train', transforms=None):
        if mode == 'train' or mode == 'val':
            self.classes = classes
            self.image_root = image_root
            self.label_root = label_root
            
            pngs = self.get_pngs()
            jsons = self.get_jsons()
            
            jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
            pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}
            
            assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
            assert len(pngs_fn_prefix - jsons_fn_prefix) == 0
            
            pngs = sorted(pngs)
            jsons = sorted(jsons)
            
            _filenames = np.array(pngs)
            _labelnames = np.array(jsons)
            
            groups = [os.path.dirname(fname) for fname in _filenames]

            ys = [0 for fname in _filenames]

            gkf = GroupKFold(n_splits=5)
            
            filenames = []
            labelnames = []
            for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
                if mode=='train':
                    if i == 0:
                        continue
                        
                    filenames += list(_filenames[y])
                    labelnames += list(_labelnames[y])
                
                else:
                    filenames = list(_filenames[y])
                    labelnames = list(_labelnames[y])

                    break
            
            self.labelnames = labelnames
            
        elif mode == 'inference':
            self.image_root = image_root
            pngs = self.get_pngs()
            filenames = np.array(sorted(pngs))

        self.filenames = filenames
        self.transforms = transforms
        self.mode = mode
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        if self.mode in ['train', 'val']:
            label_name = self.labelnames[item]
            label_path = os.path.join(self.label_root, label_name)

            label_shape = tuple(image.shape[:2]) + (len(self.classes),)
            label = np.zeros(label_shape, dtype=np.uint8)

            with open(label_path, "r") as f:
                annotations = json.load(f)["annotations"]

            phrases = [None] * len(self.classes) 

            for idx, ann in enumerate(annotations):
                c = ann["label"]
                class_ind = CLASS2IND[c]
                points = np.array(ann["points"])

                class_label = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                label[..., class_ind] = class_label  # 클래스 인덱스를 사용하여 마스크 할당

                phrases[class_ind] = c

            if self.transforms is not None:
                inputs = {"image": image, "mask": label}
                result = self.transforms(**inputs)

                image = result["image"]
                label = result["mask"]

            image = image.transpose(2, 0, 1) 
            label = label.transpose(2, 0, 1)

            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()

            target_additional = (torch.zeros(0), item)
        
            data_x = (image,) + (tuple(phrases))

            return (data_x), (label, *target_additional)

        elif self.mode == 'inference':
            # inference 시에도 text prompt 필요 
            # class 이름을 프롬포트로 사용
            phrases = self.classes

            if self.transforms is not None:
                inputs = {"image": image}
                result = self.transforms(**inputs)
                image = result["image"]
            
            image = image.transpose(2, 0, 1)  
            image = torch.from_numpy(image).float()

            data_x = (image,) + (tuple(phrases))

            return data_x, image_name

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def get_pngs(self):
        return {
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        
    def get_jsons(self):
        return {
            os.path.relpath(os.path.join(root, fname), start=self.label_root)
            for root, _dirs, files in os.walk(self.label_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
            