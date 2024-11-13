import yaml
from albumentations import (
    Compose, RandomCrop, HorizontalFlip, Rotate, Affine, Sharpen, RandomBrightnessContrast
)
import albumentations as A

def get_transform(config, is_train=True):
    aug_ops = []
    aug_ops.append(A.Resize(512, 512))

    if is_train:
        aug_dict = config['data']['train'].get('augmentation', {})
        for aug_name, aug_prob in aug_dict.items():
            if aug_name == 'crop':
                aug_ops.append(RandomCrop(150, 150, p=aug_prob))
            elif aug_name == 'flip':
                aug_ops.append(HorizontalFlip(p=aug_prob))
            elif aug_name == 'rotation':
                aug_ops.append(Rotate(limit=45, p=aug_prob))
            elif aug_name == 'affine':
                aug_ops.append(Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=30, shear=15, p=aug_prob))
            elif aug_name == 'sharpen':
                aug_ops.append(Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.5), p=aug_prob))
            elif aug_name == 'contrast':
                aug_ops.append(RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=aug_prob))

    return Compose(aug_ops)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
