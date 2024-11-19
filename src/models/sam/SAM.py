import torch
import torch.nn as nn

import torch.nn.functional as F

from .build_sam import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b

class SAM(nn.Module):
    def __init__(self, checkpoint=None) -> None:
        super(SAM, self).__init__()
        sam = build_sam_vit_b(checkpoint)
        self.sam = sam
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        
        # 목표 이미지 크기를 512로 설정
        self.img_size = 512  # 변경된 목표 이미지 크기
        self.patch_size = 16  # 패치 크기
        self.grid_size = self.img_size // self.patch_size  # 그리드 크기 조정
        
        self.mask_threshold = 0.5  # 마스크 임계값
        
        # 모델 파라미터 조정
        # self.adjust_model_parameters()
    
    def adjust_model_parameters(self):
        self.adjust_position_embeddings()
        self.adjust_relative_position_matrices()
    
    def adjust_position_embeddings(self):
        pos_embed = self.image_encoder.pos_embed
        orig_grid_size = int(pos_embed.shape[1])  # 원본 그리드 크기
        
        if orig_grid_size != self.grid_size:
            # 위치 임베딩 크기 변경: 512x512 이미지에 맞게 크기 조정
            pos_embed_resized = F.interpolate(
                pos_embed.permute(0, 3, 1, 2),  # [1, dim, H, W]
                size=(self.grid_size, self.grid_size),
                mode='bicubic',
                align_corners=False
            )
            self.image_encoder.pos_embed = nn.Parameter(pos_embed_resized.permute(0, 2, 3, 1))
            
            print(f"Position embedding resized from {orig_grid_size}x{orig_grid_size} to {self.grid_size}x{self.grid_size}")
    
    def adjust_relative_position_matrices(self):
        # Loop through all the attention blocks if you're adjusting all of them
        for block in self.sam.image_encoder.blocks:
            attn_block = block.attn

            # Calculate new size for relative position matrices
            new_size = 2 * self.grid_size - 1

            # Resize rel_pos_h
            old_rel_pos_h = attn_block.rel_pos_h
            new_rel_pos_h = F.interpolate(
                old_rel_pos_h.unsqueeze(0).unsqueeze(0),
                size=(new_size, old_rel_pos_h.size(1)),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            attn_block.rel_pos_h = nn.Parameter(new_rel_pos_h)

            # Resize rel_pos_w
            old_rel_pos_w = attn_block.rel_pos_w
            new_rel_pos_w = F.interpolate(
                old_rel_pos_w.unsqueeze(0).unsqueeze(0),
                size=(new_size, old_rel_pos_w.size(1)),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            attn_block.rel_pos_w = nn.Parameter(new_rel_pos_w)

        
    def forward(self, x):
        orig_input = x
        
        # 이미지 임베딩 얻기
        image_embeddings = self.image_encoder(x)
        
        # 마스크가 3채널(RGB)이라면 그레이스케일로 변환
        masks = x.mean(dim=1, keepdim=True)  # Convert RGB to grayscale by averaging the channels

        
        # 프롬프트 임베딩 얻기
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=masks  # 마스크 사용
        )
        
        # dense_prompt_embeddings의 크기를 image_embeddings의 크기와 맞추기
        dense_prompt_embeddings_resized = F.interpolate(
            dense_embeddings, 
            size=(image_embeddings.shape[2], image_embeddings.shape[3]), 
            mode='bilinear', 
            align_corners=False
        )

        
        # print(f"image_embeddings shape: {image_embeddings.shape}")
        # print(f"sparse_embeddings shape: {sparse_embeddings.shape}")
        # print(f"dense_embeddings shape: {dense_embeddings.shape}")
        # print(f"dense_prompt_embeddings_resized shape: {dense_prompt_embeddings_resized.shape}")


        # # 각 텐서의 채널 수 출력
        # print(f"image_embeddings shape: {image_embeddings.shape}, channels: {image_embeddings.shape[1]}")
        # print(f"sparse_embeddings shape: {sparse_embeddings.shape}, channels: {sparse_embeddings.shape[1]}")
        # print(f"dense_embeddings shape: {dense_embeddings.shape}, channels: {dense_embeddings.shape[1]}")
        # print(f"dense_prompt_embeddings_resized shape: {dense_prompt_embeddings_resized.shape}, channels: {dense_prompt_embeddings_resized.shape[1]}")
        # print(f"prompt_encoder.get_dense_pe shape: {self.prompt_encoder.get_dense_pe().shape}")
    
        
        # 마스크 디코딩
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings_resized,
            multimask_output=True,
        )
        
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=low_res_masks.shape[-2:],
            original_size=orig_input.shape[-2:]
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