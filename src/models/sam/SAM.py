import torch
import torch.nn as nn

import torch.nn.functional as F

from .build_sam import sam_model_registry

import os
import subprocess

from typing import List


def download_file(url, save_path):
    if not os.path.exists(save_path):
        print(f"File not found at {save_path}. Downloading from {url}...")
        try:
            subprocess.run(['curl', '-o', save_path, url], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download the file from {url}. Error: {e}")


def get_sam(model_name, hiera_dir='./sam_pretrained'):
    os.makedirs(os.path.dirname(hiera_dir), exist_ok=True)
    model_sizes = {
        "b": ("sam_vit_b.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
        "h": ("sam_vit_h.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
        "l": ("sam_vit_l.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
    }

    model_size = model_name.split('_')[1]
    if model_size not in model_sizes:
        raise ValueError(f"Unknown SAM2UNet size: sam_vit_{model_size}")

    hiera_file, download_url = model_sizes[model_size]
    os.makedirs(hiera_dir, exist_ok=True)
    hiera_path = os.path.join(hiera_dir, hiera_file)

    download_file(download_url, hiera_path)

    return SAM(hiera_path)


class SAM(nn.Module):
    def __init__(self,
                 checkpoint=None,
                 pixel_mean: List[float] = [123.675, 116.28, 103.53],
                 pixel_std: List[float] = [58.395, 57.12, 57.375],
        ) -> None:
        super(SAM, self).__init__()
        sam = sam_model_registry['vit_b'](checkpoint='./sam_pretrained/sam_vit_b.pth')
        self.sam = sam
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        
        
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        
        # 목표 이미지 크기를 512로 설정
        # self.img_size = 512  # 변경된 목표 이미지 크기
        # self.patch_size = 16  # 패치 크기
        # self.grid_size = self.img_size // self.patch_size  # 그리드 크기 조정
        
        self.mask_threshold = 0.5  # 마스크 임계값
        
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        

        
    def forward(self, img, mask):
        orig_input_img = img      # shape = (2, 3, 512, 512)
        orig_input_mask = mask    # shape = (2, 1, 512, 512)
        
        input_images = self.preprocess(img)      # shape = (2, 3, 1024, 1024)
        image_embeddings = self.image_encoder(input_images)      # shape = (2, 256, 64, 64)
        
        
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=mask  # 마스크 사용
        )

        
        # 마스크 디코딩
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        
        
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=low_res_masks.shape[-2:],
            original_size=orig_input_img.shape[-2:]
        )
        
        
        # masks = masks > self.mask_threshold
        # sigmoid = torch.nn.Sigmoid()
        # masks = sigmoid(masks)
        
        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: tuple[int, ...],
        original_size: tuple[int, ...],
        ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
    
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        batch_size, c, h, w = x.shape
    
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        
        return x
