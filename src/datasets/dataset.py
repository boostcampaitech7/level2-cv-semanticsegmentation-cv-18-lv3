# python native
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from src.utils.augmentation import get_transform

from typing import Union

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
            
            self.filenames = _filenames
            self.labelnames = _labelnames
            
            # split train-valid
            # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
            # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
            # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
            groups = [os.path.dirname(fname) for fname in _filenames]
            
            # dummy label
            ys = [0 for fname in _filenames]
            
            # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
            # 5으로 설정하여 KFold를 수행합니다.
            gkf = GroupKFold(n_splits=5)
            
            filenames = []
            labelnames = []
            for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
                if mode=='train':
                    # 0번을 validation dataset으로 사용합니다.
                    if i == 0:
                        continue
                        
                    filenames += list(_filenames[y])
                    labelnames += list(_labelnames[y])
                
                else:
                    filenames = list(_filenames[y])
                    labelnames = list(_labelnames[y])
                    
                    # skip i > 0
                    break
            
            self.labelnames = labelnames
            
        elif mode == 'inference':
            self.image_root = image_root
            pngs = self.get_pngs()
            filenames = np.array(sorted(pngs))

        self.filenames = filenames
        self.transforms = transforms
        self.mode = mode
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, item: int) -> Union[tuple[Tensor, Tensor], tuple[Tensor, str]]:
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.mode == 'train' or self.mode == 'val':
            label_name = self.labelnames[item]
            label_path = os.path.join(self.label_root, label_name)
            
            # (H, W, NC) 모양의 label을 생성합니다.
            label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
            label = np.zeros(label_shape, dtype=np.uint8)
            
            # label 파일을 읽습니다.
            with open(label_path, "r") as f:
                annotations = json.load(f)
            annotations = annotations["annotations"]
            
            # 클래스 별로 처리합니다.
            for ann in annotations:
                c = ann["label"]
                class_ind = CLASS2IND[c]
                points = np.array(ann["points"])
                
                # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
                class_label = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                label[..., class_ind] = class_label
            
            if self.transforms is not None:
                inputs = {"image": image, "mask": label} if self.mode == 'train' else {"image": image}
                result = self.transforms(**inputs)
                
                image = result["image"]
                label = result["mask"] if self.mode == 'train' else label

            # to tenser will be done later
            image = image.transpose(2, 0, 1)    # channel first 포맷으로 변경합니다.
            label = label.transpose(2, 0, 1)
            
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()
                
            return image, label
        
        elif self.mode == 'inference':
            image_name = self.filenames[item]
            image_path = os.path.join(self.image_root, image_name)
            
            image = cv2.imread(image_path)
            image = image / 255.
            
            if self.transforms is not None:
                inputs = {"image": image}
                result = self.transforms(**inputs)
                image = result["image"]

            # to tenser will be done later
            image = image.transpose(2, 0, 1)  
            
            image = torch.from_numpy(image).float()
                
            return image, image_name
    
    def get_pngs(self) -> set[str]:
        return {
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        
    def get_jsons(self) -> set[str]:
        return {
            os.path.relpath(os.path.join(root, fname), start=self.label_root)
            for root, _dirs, files in os.walk(self.label_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
        
    def get_filepath(self, idx: int) -> tuple[str, str]:
        return self.filenames[idx], self.labelnames[idx]
    
    def get_all_path(self) -> tuple[list, list]:
        return self.filenames, self.labelnames
            
if __name__ == "__main__":

    img_root = "data/train/DCM"
    label_root = "data/train/outputs_json"
    test_img_root = "data/test/DCM"
    
    temp_train = XRayDataset(img_root, label_root, CLASSES, 'train', None)
    temp_val = XRayDataset(img_root, label_root, CLASSES, 'val', None)
    temp_test = XRayDataset(img_root, None, CLASSES, 'test', None)
    
    print(temp_train[0])
    print(temp_val[0])
    print(temp_test[0])