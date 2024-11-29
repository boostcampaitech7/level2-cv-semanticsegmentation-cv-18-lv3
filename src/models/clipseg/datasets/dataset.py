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

BONE_ANATOMICAL_DETAILS = {
    'finger-1': "The distal phalanx of the thumb is the most distal bone, responsible for supporting the nail bed and enabling fine motor precision in the thumb.",
    'finger-2': "The distal phalanx of the index finger is the terminal bone that facilitates precise pointing and grasping functions.",
    'finger-3': "The distal phalanx of the middle finger is the end bone, supporting the fingertip for precise manipulative tasks.",
    'finger-4': "The distal phalanx of the ring finger is the terminal bone contributing to grip strength and finger extension.",
    'finger-5': "The distal phalanx of the pinky finger provides structural support to the fingertip and assists in grip and manipulation tasks.",
    'finger-6': "The middle phalanx of the index finger connects the distal and proximal phalanges, enabling flexion and extension of the finger.",
    'finger-7': "The middle phalanx of the middle finger provides leverage for flexion and extension and is critical for strong grip functions.",
    'finger-8': "The middle phalanx of the ring finger facilitates bending and straightening motions in the finger.",
    'finger-9': "The middle phalanx of the pinky finger supports fine motor adjustments during gripping.",
    'finger-10': "The proximal phalanx of the thumb forms the base of the thumb and is essential for opposability and dexterity.",
    'finger-11': "The proximal phalanx of the index finger serves as a pivot point for finger movement and contributes to precision grip.",
    'finger-12': "The proximal phalanx of the middle finger forms the primary bone structure for forceful grip and extension.",
    'finger-13': "The proximal phalanx of the ring finger is a key structure in the finger that aids in flexion and extension.",
    'finger-14': "The proximal phalanx of the pinky finger allows for articulation and movement in the small finger, assisting in overall hand flexibility.",
    'finger-15': "The metacarpal of the thumb supports the base of the thumb and contributes to thumb mobility and stability during gripping.",
    'finger-16': "The metacarpal of the index finger provides structural support for the index finger and is integral to strong pinch grips.",
    'finger-17': "The metacarpal of the middle finger is the central support for the hand and facilitates force transmission during grasping.",
    'finger-18': "The metacarpal of the ring finger provides structural support to the ring finger, aiding in grip strength and balance.",
    'finger-19': "The metacarpal of the pinky finger stabilizes the small finger and enhances flexibility for intricate hand movements.",
    'Trapezium': "The trapezium is a carpal bone that articulates with the first metacarpal, enabling thumb movement and opposition.",
    'Trapezoid': "The trapezoid is a small, wedge-shaped carpal bone that supports the base of the index finger.",
    'Capitate': "The capitate is the largest carpal bone and acts as the central anchor for the carpal bones in the wrist.",
    'Hamate': "The hamate is a wedge-shaped carpal bone that features a hook-like process for ligament attachment.",
    'Scaphoid': "The scaphoid is a carpal bone that bridges the proximal and distal rows of carpal bones, playing a key role in wrist stability.",
    'Lunate': "The lunate is a crescent-shaped carpal bone that enables wrist flexion and extension.",
    'Triquetrum': "The triquetrum is a three-sided carpal bone that forms the ulnar side of the wrist joint.",
    'Pisiform': "The pisiform is a small, pea-shaped carpal bone embedded in a tendon, serving as a leverage point for wrist movement.",
    'Radius': "The radius is one of the two long bones in the forearm, playing a crucial role in forearm rotation and wrist joint articulation.",
    'Ulna': "The ulna is a long bone in the forearm that forms the primary articulation with the humerus at the elbow joint."

BONE_LOCATE_CONTEXT = {
    'finger-1': "This bone is the distal phalanx of the thumb, which is located at the tip of the thumb.",
    'finger-2': "This bone is the distal phalanx of the index finger, which is located at the tip of the index finger.",
    'finger-3': "This bone is the distal phalanx of the middle finger, which is located at the tip of the middle finger.",
    'finger-4': "This bone is the distal phalanx of the ring finger, which is located at the tip of the ring finger.",
    'finger-5': "This bone is the distal phalanx of the pinky finger, which is located at the tip of the pinky finger.",
    'finger-6': "This bone is the middle phalanx of the index finger, which is located between the proximal and distal phalanges of the index finger.",
    'finger-7': "This bone is the middle phalanx of the middle finger, which is located between the proximal and distal phalanges of the middle finger.",
    'finger-8': "This bone is the middle phalanx of the ring finger, which is located between the proximal and distal phalanges of the ring finger.",
    'finger-9': "This bone is the middle phalanx of the pinky finger, which is located between the proximal and distal phalanges of the pinky finger.",
    'finger-10': "This bone is the proximal phalanx of the thumb, which is located at the base of the thumb.",
    'finger-11': "This bone is the proximal phalanx of the index finger, which is located at the base of the index finger.",
    'finger-12': "This bone is the proximal phalanx of the middle finger, which is located at the base of the middle finger.",
    'finger-13': "This bone is the proximal phalanx of the ring finger, which is located at the base of the ring finger.",
    'finger-14': "This bone is the proximal phalanx of the pinky finger, which is located at the base of the pinky finger.",
    'finger-15': "This bone is the metacarpal of the thumb, which is located in the palm beneath the thumb.",
    'finger-16': "This bone is the metacarpal of the index finger, which is located in the palm beneath the index finger.",
    'finger-17': "This bone is the metacarpal of the middle finger, which is located in the palm beneath the middle finger.",
    'finger-18': "This bone is the metacarpal of the ring finger, which is located in the palm beneath the ring finger.",
    'finger-19': "This bone is the metacarpal of the pinky finger, which is located in the palm beneath the pinky finger.",
    'Trapezium': "This bone is the trapezium, which is located at the base of the thumb in the wrist.",
    'Trapezoid': "This bone is the trapezoid, which is located in the wrist beneath the index finger.",
    'Capitate': "This bone is the capitate, which is located at the center of the wrist.",
    'Hamate': "This bone is the hamate, which is located in the wrist near the bases of the ring and pinky fingers.",
    'Scaphoid': "This bone is the scaphoid, which is located in the wrist near the base of the thumb.",
    'Lunate': "This bone is the lunate, which is located at the center of the wrist near the scaphoid and triquetrum.",
    'Triquetrum': "This bone is the triquetrum, which is located on the ulnar side of the wrist near the pisiform.",
    'Pisiform': "This bone is the pisiform, which is located on the ulnar side of the wrist above the triquetrum.",
    'Radius': "This bone is the radius, which is located on the thumb side of the forearm.",
    'Ulna': "This bone is the ulna, which is located on the pinky side of the forearm."
}

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

                phrases[class_ind] = BONE_ANATOMICAL_DETAILS[c]
                # phrases[class_ind] = BONE_LOCATE_CONTEXT[c]

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
            phrases = CLASSES

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
            