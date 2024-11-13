import torch
import torch.nn as nn

class ConvBlock(nn.Module): # 기본적으로 사용할 3x3 Conv Block 정의
    expansion = 1  

    def __init__(self, in_channels: int, out_channels: int, stride: int=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None #만약 어떤 블럭에서 stride가 2 이상이라면, input(x)와 block 거치고 나온 크기가 달라서 
        #skip connection이 불가능해진다. 따라서 downsample을 해줘야함 그 방식은 1x1 Conv를 적용하는 것(참고 : https://velog.io/@gibonki77/ResNetwithPyTorch) 
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity 
        out = self.relu(out)

        return out