import yaml
import ttach as tta

tta_list = {
    "HorizontalFlip": tta.HorizontalFlip,
    "Scale": tta.Scale
}

def get_tta_transform(tta_config, is_train=False) :
    transforms = []

    if not is_train:
        tta_config = tta_config['inference'].get('tta', {})
        for transform_name, transform_params in tta_config.items():
            if transform_name in tta_list:
                transforms.append(tta_list[transform_name](**transform_params))

    for tta_name, tta_params in tta_config.get('base_tta', {}).items():
        if tta_name in tta_list:
            transforms.append(tta_list[tta_name](**tta_params))

    return tta.Compose(transforms)

def load_config(config_path) :
    with open(config_path, 'r') as file :
        return yaml.safe_load(file)