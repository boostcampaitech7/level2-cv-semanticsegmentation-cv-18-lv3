import torch
import torch.nn as nn
from .resnet34 import ResNet34
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super(DecoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # scale_factor가 2일 때 size가 2배가 되는 듯 함 
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
    
        return x

class UNetResNet34(nn.Module):
    def __init__(self, num_classes: int=29):
        super(UNetResNet34, self).__init__()
        self.encoder = ResNet34()  
        
        #skip connection은 x5랑 x4 / d4랑 x3 / d3이랑 x2 /d2랑 x1이 이루어짐.
        
        self.decoder4 = DecoderBlock(512, 256, 256) #x5의 channel은 512, x4는 256이니까 이렇게 합치고 결과로 256이 됨
        self.decoder3 = DecoderBlock(256, 128, 128) #마찬가지로 이전 결과 256과 x3 128
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1) #마지막은 64에서 classes로

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)

        d4 = self.decoder4(x5, x4)  # 32x32x512 -> 64x64x256
        d3 = self.decoder3(d4, x3)  # 64x64x256 -> 128x128x128
        d2 = self.decoder2(d3, x2)  # 128x128x128 -> 256x256x64 
        d1 = self.decoder1(d2, x1)  # 256x256x64 -> 512x512xnum_classes

        # 최종 출력
        return self.final_conv(d1)