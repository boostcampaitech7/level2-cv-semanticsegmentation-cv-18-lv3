import torch
import torch.nn as nn
import torch.nn.functional as F

 
def conv_block(in_ch, out_ch, k_size, stride, padding, dilation=1, relu=True):
    block = []
    block.append(nn.Conv2d(in_ch, out_ch, k_size, stride, padding, dilation, bias=False))
    block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())
    return nn.Sequential(*block)

class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, downsample=False):
        super().__init__()
        # Bottleneck 구조: 1x1 Conv -> 3x3 Conv -> 1x1 Conv
        self.block = nn.Sequential(
            conv_block(in_ch, out_ch//4, 1, 1, 0),
            conv_block(out_ch//4, out_ch//4, 3, stride, dilation, dilation),
            conv_block(out_ch//4, out_ch, 1, 1, 0, relu=False)
        )
        self.downsample = nn.Sequential(
            conv_block(in_ch, out_ch, 1, stride, 0, 1, False)
        ) if downsample else None

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x   # skip connection
        out = self.block(x) # bottleneck 블록

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dilation, num_layers):
        super().__init__()
        block = [] # Bottleneck 블록을 담을 리스트
        for i in range(num_layers):
            block.append(Bottleneck(
                in_ch if i==0 else out_ch, # 첫 번째는 입력 채널 유지
                out_ch,
                stride if i==0 else 1,  # 첫 번째는 stride 적용
                dilation,               # 모든 레이어에 dilation 동일하게 적용 
                True if i==0 else False # 첫 번째는 다운샘플링 필요
            ))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class ResNet101(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.conv1 = nn.Sequential(
            conv_block(in_channels, 64, 7, 2, 3), # 7x7 Conv
            nn.MaxPool2d(3, 2, 1)                 # 3x3 MaxPooling
        )
        self.layer1 = ResBlock(64, 256, 1, 1, 3)
        self.layer2 = ResBlock(256, 512, 2, 1, 4)
        self.layer3 = ResBlock(512, 1024, 1, 2, 23)
        self.layer4 = ResBlock(1024, 2048, 1, 4, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


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

def DeepLabV3_scratch(in_channels, num_classes):
    return DeepLabV3(in_channels, num_classes)