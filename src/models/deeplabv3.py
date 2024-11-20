import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet101 import ResNet101, conv_block

class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        # ASPP
        self.conv1 = conv_block(in_ch, out_ch, 1, 1, 0)
        self.conv2 = conv_block(in_ch, out_ch, 3, 1, 6, 6)   # rate = 6
        self.conv3 = conv_block(in_ch, out_ch, 3, 1, 12, 12) # rate = 12
        self.conv4 = conv_block(in_ch, out_ch, 3, 1, 18, 18) # rate = 18        
        self.conv5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            conv_block(in_ch, out_ch, 1, 1, 0)
        )

        self.conv6 = conv_block(out_ch * 5, out_ch, 1, 1, 0)

    def forward(self, x):
        # original image size
        size = x.shape[2:]
        
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x5 = F.interpolate(x5, size=size, mode="bilinear", align_corners=False)  # Upsample global feature
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1) # 모든 출력 텐서를 채널 방향으로 연결
        x = self.conv6(x)
        return x


class DeepLabV3(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV3, self).__init__()

        # Backbone
        self.backbone = ResNet101(in_channels)

        # ASPP
        self.aspp = AtrousSpatialPyramidPooling(2048, 256)

        # Decoder
        self.low_level_conv = conv_block(256, 48, 1, 1, 0)
        self.decoder = nn.Sequential(
            conv_block(304, 256, 3, 1, 1), # Combine ASPP + low-level features
            conv_block(256, 256, 3, 1, 1),
            nn.Conv2d(256, num_classes, kernel_size=1) # Final 1x1 Conv for segmentation map
        )

    def forward(self, x):
        # original image size
        size = x.shape[2:]
        
        # Backbone
        conv1_feat = self.backbone.conv1(x)  # conv1 output
        low_level_feat = self.backbone.layer1(conv1_feat) # low-level feature 추출
        x = self.backbone.layer2(low_level_feat)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # ASPP
        x = self.aspp(x)
        
        # Decoder
        low_level_feat = self.low_level_conv(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)

        # Upsample
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x
