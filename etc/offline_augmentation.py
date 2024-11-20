import os
import json
from tqdm import tqdm
import shutil

import cv2
import numpy as np
import albumentations as A
from albumentations.core.serialization import Serializable

from typing import List, Tuple

from src.datasets.dataset import XRayDataset, CLASSES, CLASS2IND
from src.utils.augmentation import get_transform

IMAGE_ROOT = "data/train/DCM"
LABEL_ROOT = "data/train/outputs_json"
AUGMENTED_IMAGE_ROOT = "data/train/DCM_augmented"
AUGMENTED_LABEL_ROOT = "data/train/outputs_json_augmented"
ALL_IMAGE_ROOT = "data/train/DCM_all"
ALL_LABEL_ROOT = "data/train/outputs_json_all"
NUM_AUGMENTATIONS = 1  

os.makedirs(AUGMENTED_IMAGE_ROOT, exist_ok=True)
os.makedirs(AUGMENTED_LABEL_ROOT, exist_ok=True)

def create_mask_from_polygons(
    polygons: List[List[List[int]]],
    class_indices: List[int],
    height: int,
    width: int
) -> np.ndarray:

    #polygon list를 통해서 mask 생성 
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon, class_idx in zip(polygons, class_indices):
        pts = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(mask, pts, class_idx + 1)  
    return mask

def extract_polygons_from_mask(mask: np.ndarray) -> Tuple[List[List[List[int]]], List[int]]:

    #mask 에서 polygon 추출 
    polygons = []
    class_indices = []
    # 클래스별로 루프를 돌며 폴리곤 추출
    for class_idx in np.unique(mask):
        if class_idx == 0:
            # 배경 skip
            continue  
        mask_class = (mask == class_idx).astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            mask_class,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            polygon = contour.reshape(-1, 2).tolist()
            if len(polygon) >= 3:  
                polygons.append(polygon)
                class_indices.append(class_idx - 1) 
    return polygons, class_indices

def get_offline_transform() -> A.Compose:
    transform = A.Compose(
        [
            A.Sharpen(
                alpha=(0.2, 0.5),  # 샤픈 강도 범위
                lightness=(0.8, 1.2),  # 밝기 조정 범위
                p=1  # 적용 확률
            ),
        ]
    )
    return transform


def load_annotations(label_path: str) -> List[dict]:
    #json file 에서 annotatation 로드 
    with open(label_path, "r") as f:
        data = json.load(f)
    annotations = data["annotations"]
    return annotations

def save_augmented_image(image: np.ndarray, image_name: str) -> None:
    #증강된 이미지 저장 
    aug_image_path = os.path.join(AUGMENTED_IMAGE_ROOT, image_name)
    aug_image_dir = os.path.dirname(aug_image_path)
    os.makedirs(aug_image_dir, exist_ok=True)

    image_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(aug_image_path, image_bgr)

def save_augmented_label(
    polygons: List[List[List[int]]],
    class_indices: List[int],
    label_name: str
) -> None:
    #증강된 라벨을 json 형식으로 저장
    annotations = []
    for polygon, class_idx in zip(polygons, class_indices):
        annotations.append({
            "label": CLASSES[class_idx],
            "points": polygon
        })

    aug_label_path = os.path.join(AUGMENTED_LABEL_ROOT, label_name)
    aug_label_dir = os.path.dirname(aug_label_path)
    os.makedirs(aug_label_dir, exist_ok=True)

    with open(aug_label_path, "w") as f:
        json.dump({"annotations": annotations}, f, indent=4)

def merge_folders(folder1: str, folder2: str, destination_folder: str) -> None:
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for root, dirs, files in os.walk(folder1):
        for file in files:
            relative_path = os.path.relpath(root, folder1)
            dest_dir = os.path.join(destination_folder, relative_path)
            os.makedirs(dest_dir, exist_ok=True)

            src_path = os.path.join(root, file)
            dest_path = os.path.join(dest_dir, file)
            shutil.copy(src_path, dest_path)

    for root, dirs, files in os.walk(folder2):
        for file in files:
            relative_path = os.path.relpath(root, folder2)
            dest_dir = os.path.join(destination_folder, relative_path)
            os.makedirs(dest_dir, exist_ok=True)

            src_path = os.path.join(root, file)
            dest_path = os.path.join(dest_dir, file)

            if os.path.exists(dest_path):
                base, ext = os.path.splitext(file)
                dest_path = os.path.join(dest_dir, f"{base}_copy{ext}")
            shutil.copy(src_path, dest_path)

    print(f"Files merged into {destination_folder}")

def main() -> None:
    transform = get_offline_transform()

    dataset = XRayDataset(
        image_root=IMAGE_ROOT,
        label_root=LABEL_ROOT,
        classes=CLASSES,
        mode='train',
        transforms=None  
    )

    for idx in tqdm(range(len(dataset)), desc="Augmenting data"):
        image, label = dataset[idx] 

        image_np = image.numpy().transpose(1, 2, 0) 
        height, width = image_np.shape[:2]

        label_name = dataset.labelnames[idx]
        annotations = load_annotations(os.path.join(LABEL_ROOT, label_name))

        polygons = []
        class_indices = []
        for ann in annotations:
            polygons.append(ann["points"])
            class_indices.append(CLASS2IND[ann["label"]])

        mask = create_mask_from_polygons(
            polygons=polygons,
            class_indices=class_indices,
            height=height,
            width=width
        )

        for aug_idx in range(NUM_AUGMENTATIONS):
            try:
                augmented = transform(image=image_np, mask=mask)
                aug_image = augmented["image"]
                aug_mask = augmented["mask"]
            except Exception as e:
                print(f"Augmentation failed for index {idx} with error: {e}")
                continue

            # 증강된 마스크에서 폴리곤 추출
            aug_polygons, aug_class_indices = extract_polygons_from_mask(aug_mask)

            # 증강된 데이터 파일명 설정
            image_name = dataset.filenames[idx]
            base_name = os.path.splitext(image_name)[0]
            aug_image_name = f"{base_name}_aug{aug_idx}.png"
            aug_label_name = f"{base_name}_aug{aug_idx}.json"

            # 이미지 저장
            save_augmented_image(
                image=aug_image,
                image_name=aug_image_name
            )

            # 라벨 저장
            save_augmented_label(
                polygons=aug_polygons,
                class_indices=aug_class_indices,
                label_name=aug_label_name
            )

    print("Offline augmentation 완료.")
    merge_folders(IMAGE_ROOT, AUGMENTED_IMAGE_ROOT, ALL_IMAGE_ROOT)
    merge_folders(LABEL_ROOT, AUGMENTED_LABEL_ROOT, ALL_LABEL_ROOT)

if __name__ == "__main__":
    main()
