import yaml
import torch
import torch.nn as nn
from typing import Optional, Mapping, Union, Tuple

import ttach as tta

class SegmentationTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (segmentation model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): segmentation model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `mask`
    """

    def __init__(
        self,
        model: nn.Module,
        transforms: tta.Compose,
        merge_mode: str = "mean",
        output_mask_key: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_mask_key
    
    def forward(
        self, image: torch.Tensor, *args
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        merger = tta.base.Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)

            if isinstance(augmented_output, tuple):
                augmented_output = augmented_output[0]            
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_mask(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result

tta_list = {
    "HorizontalFlip": tta.HorizontalFlip,
    "VerticalFlip": tta.VerticalFlip,
    "Rotate90": tta.Rotate90,
    "Scale": tta.Scale,
    "Resize": tta.Resize,
    "Add": tta.Add,
    "Multiply": tta.Multiply,
    "FiveCrops": tta.FiveCrops,
}

def get_tta_transform(tta_config, is_train=False) :
    transforms = []

    if not is_train:
        tta_config = tta_config['inference'].get('tta', {})
        for transform_name, transform_params in tta_config.items():
            if transform_name in tta_list:
                if transform_params is None:
                    transforms.append(tta_list[transform_name]())
                else:
                    transforms.append(tta_list[transform_name](**transform_params))

    # transforms 리스트 출력
    print("Transforms List:", transforms)
    
    return tta.Compose(transforms)

def load_config(config_path) :
    with open(config_path, 'r') as file :
        return yaml.safe_load(file)