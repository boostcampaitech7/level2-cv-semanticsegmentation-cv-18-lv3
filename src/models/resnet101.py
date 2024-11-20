import torch
import torch.nn as nn

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