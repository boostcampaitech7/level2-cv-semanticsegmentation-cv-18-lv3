import os
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cv2
from PIL import Image

class VisualizeMetaData():
    def __init__(self, xlsx_path) -> None:
        self.xlsx_path = xlsx_path
        
        self.meta_data = self.load_xlsx(xlsx_path)
    
    @staticmethod
    def load_xlsx(file_path: str) -> pd.DataFrame:
        meta_data = pd.read_excel(file_path)
        
        meta_data = meta_data.drop(columns=['ID', 'Unnamed: 5'])
        meta_data['성별'] = meta_data['성별'].str.replace('_x0008_','',regex=False).str.strip()
        meta_data['성별'] = meta_data['성별'].replace({'남': 'Male', '여': 'Female'})
        
        return meta_data
    
    def plot_preview(self) -> None:
        st.subheader("Data Preview")
        st.write(self.meta_data.head())

        st.subheader("Descriptive Statistics")
        st.write(self.meta_data.describe().T)
    
    def plot_sex_distribution(self) -> None:
        plt.figure(figsize=(5, 5))
        sns.countplot(x=self.meta_data['성별'])
        plt.title('Sex Distribution')
        plt.xlabel('Sex')
        st.pyplot(plt)
    
    def plot_histogram(self) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sns.histplot(self.meta_data['나이'], kde=True, ax=axes[0])
        axes[0].set_title('Age Distribution')
        axes[0].set_xlabel('Age')

        sns.histplot(self.meta_data['체중(몸무게)'], kde=True, ax=axes[1])
        axes[1].set_title('Weight Distribution')
        axes[1].set_xlabel('Weight')

        sns.histplot(self.meta_data['키(신장)'], kde=True, ax=axes[2])
        axes[2].set_title('Height Distribution')
        axes[2].set_xlabel('Height')

        st.pyplot(fig)
    
class VisualizeImageAndAnnotation():
    def __init__(self, img_root, label_root, test_root, csv_path=None) -> None:
        self.img_root = img_root
        self.label_root = label_root
        self.test_root = test_root
        self.csv_path = csv_path
        
        self.classes = [
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]
        
        self.palette = [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
            (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
            (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
            (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
        ]
        
        self.class2ind = {v: i for i, v in enumerate(self.classes)}
        self.ind2class = {v: k for k, v in self.class2ind.items()}
        
        self.pngs = sorted(self.load_pngs(img_root))
        self.jsons = sorted(self.load_jsons(label_root))
        self.tests = sorted(self.load_pngs(test_root))
        
        self.img_to_path = {path.split('/')[1]: path for path in self.tests}
        
        if csv_path:
            self.csv_pd = self.load_csv(csv_path, self.img_to_path)
            
        self.xraydataset = XRayDataset(
                                self.pngs, 
                                self.jsons, 
                                self.img_root,
                                self.label_root,
                                self.classes,
                                self.class2ind)
        
    @staticmethod
    def load_pngs(image_root: str) -> set:
        return {
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _dirs, files in os.walk(image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
    
    @staticmethod
    def load_jsons(label_root: str) -> set:
        return {
            os.path.relpath(os.path.join(root, fname), start=label_root)
            for root, _dirs, files in os.walk(label_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
    
    @staticmethod
    def load_csv(csv_path: str, mapping: dict) -> pd.DataFrame:
        csv_pd = pd.read_csv(csv_path)
        
        csv_pd['image_name'] = csv_pd['image_name'].map(mapping)
        
        return csv_pd
    
    def set_csv(self, csv_path: str) -> None:
        self.csv_pd = self.load_csv(csv_path, self.img_to_path)
    
    def get_train_count(self):
        return len(self.jsons)
    
    def get_test_count(self):
        return len(self.tests)
    
    def get_json_info(self, json_path):
        pass
    
    def plot_base_img(self, idx: int):
        img_path = os.path.join(self.img_root, self.pngs[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        st.image(image, caption=f"{self.pngs[idx]}")
    
    def plot_gt_and_pred(self):
        pass
    
    def plot_train_annotation(self, idx: int): 
        image, label = self.xraydataset[idx]
        image = image.numpy().transpose(1, 2, 0)
        label = label.numpy()
        
        image = (image * 255).astype(np.uint8)
        
        label_rgb = self.label2rgb(label)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(image)
        ax.imshow(label_rgb, alpha=0.3)  # label_rgb에 투명도 설정하여 겹치기
        ax.axis("off")
        
        # Streamlit에 이미지 표시
        st.pyplot(fig)
        
    
    def plot_pred_only(self, idx: int):
        grouped = self.csv_pd.groupby("image_name")
        
        image_path = os.path.join(self.test_root, self.tests[idx])
        
        image = np.array(Image.open(image_path).convert('RGB'))
        height, width = image.shape[:2]
        mask = np.zeros((len(self.palette), height, width), dtype=np.uint8)
        
        for _, row in grouped.get_group(self.tests[idx]).iterrows():
            class_name, rle = row['class'], row['rle']
            decoded_mask = self.rle_decode(rle, (height, width))
            mask[self.class2ind[class_name]] = decoded_mask
        
        color_mask = self.label2rgb(mask, height, width)
        blended = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)
        blended = Image.fromarray(blended)
        st.image(blended, caption=f"{self.tests[idx]}")

    def label2rgb(self, label):
        image_size = label.shape[1:] + (3, )
        image = np.zeros(image_size, dtype=np.uint8)
        
        for i, class_label in enumerate(label):
            image[class_label == 1] = self.palette[i]
            
        return image
    
    def label2rgb(self, label: np.ndarray, height: int, width: int) -> np.ndarray:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i, class_mask in enumerate(label):
            image[class_mask == 1] = self.palette[i]
        return image
    
    @staticmethod
    def rle_decode(mask_rle: str, shape: tuple[int, int]) -> np.ndarray:
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)


class XRayDataset(Dataset):
    def __init__(self, pngs: list, jsons: list, img_root: str, label_root: str, classes: list, class2ind: dict, transforms=None):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        self.img_root = img_root
        self.label_root = label_root
        self.classes = classes
        self.class2ind = class2ind
        
        self.filenames = list(_filenames)
        self.labelnames = list(_labelnames)

        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.img_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        label_shape = tuple(image.shape[:2]) + (len(self.classes), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        for ann in annotations:
            c = ann["label"]
            class_ind = self.class2ind[c]
            points = np.array(ann["points"])
            
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        image = image.transpose(2, 0, 1) 
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label
        

if __name__=="__main__":
    img_root = "/data/ephemeral/home/kwak/level2-cv-semanticsegmentation-cv-18-lv3/data/train/DCM"
    label_root = "/data/ephemeral/home/kwak/level2-cv-semanticsegmentation-cv-18-lv3/data/train/outputs_json"
    test_root = "/data/ephemeral/home/kwak/level2-cv-semanticsegmentation-cv-18-lv3/data/test/DCM"
    csv_path = "/data/ephemeral/home/kwak/level2-cv-semanticsegmentation-cv-18-lv3/outputs/saved_models/temp_2020/output_baseline_100ep.csv"
    a = VisualizeImageAndAnnotation(img_root, label_root, test_root, csv_path)
    a.plot_pred_only(0)
    
