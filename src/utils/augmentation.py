import yaml
from albumentations import (
    Resize, Normalize, Compose, RandomCrop, HorizontalFlip, Rotate, Affine, Sharpen, RandomBrightnessContrast
)
import albumentations as A

from typing import Any

aug_list = {
    'resize' : Resize,
    'normalize': Normalize,
    'crop' : RandomCrop,
    'hflip' : HorizontalFlip,
    'rotate' : Rotate,
    'affine' : Affine,
    'sharpen' : Sharpen,
    'contrast' : RandomBrightnessContrast
}

def get_transform(aug_config: dict[str, Any], is_train: bool=True) -> Compose:
    aug_ops = []

    if is_train:
        train_aug_config = aug_config['train'].get('augmentation', {})
        for aug_name, aug_params in train_aug_config.items():
            aug_ops.append(aug_list[aug_name](**aug_params))

    for aug_name, aug_params in aug_config['base_augmentation'].items():
        aug_ops.append(aug_list[aug_name](**aug_params))

    return Compose(aug_ops)

def load_config(config_path: str) -> None:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)