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
        for aug_name, aug_params in aug_dict.items():
            prob = aug_params.get('prob', 1)

            if aug_name == 'crop':
                height = aug_params.get('height', 150)
                width = aug_params.get('width', 150)
                aug_ops.append(RandomCrop(height=height, width=width, p=prob))
            
            elif aug_name == 'flip':
                aug_ops.append(HorizontalFlip(p=prob))

            elif aug_name == 'rotation':
                limit = aug_params.get('limit', 45)
                aug_ops.append(Rotate(limit=limit, p=prob))

            elif aug_name == 'affine':
                scale = tuple(aug_params.get('scale', [0.9, 1.1]))
                translate_percent = tuple(aug_params.get('translate_percent', [0.1, 0.1]))
                rotate = aug_params.get('rotate', 30)
                shear = aug_params.get('shear', 15)
                aug_ops.append(Affine(scale=scale, translate_percent=translate_percent, rotate=rotate, shear=shear, p=prob))

            elif aug_name == 'sharpen':
                alpha = tuple(aug_params.get('alpha', [0.2, 0.5]))
                lightness = tuple(aug_params.get('lightness', [0.5, 1.5]))
                aug_ops.append(Sharpen(alpha=alpha, lightness=lightness, p=prob))
            
            elif aug_name == 'contrast':
                brightness_limit = aug_params.get('brightness_limit', 0.3)
                contrast_limit = aug_params.get('contrast_limit', 0.3)
                aug_ops.append(RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=prob))

    return Compose(aug_ops)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
