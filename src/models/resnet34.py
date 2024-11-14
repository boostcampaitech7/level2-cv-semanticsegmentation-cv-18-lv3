import torch
import torch.nn as nn

class ConvBlock(nn.Module): # 기본적으로 사용할 3x3 Conv Block 정의

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
        #x가 변하니 미리 저장
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
            
        x += identity 
        x = self.relu(x)

        return x
    
    
class LayerBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, repeat: int, stride=1):
        super(LayerBlock, self).__init__()
        layers = []
        # 첫 블록은 크기가 줄어듬(첫 레이어 빼곤 stride=2)
        layers.append(ConvBlock(in_channels, out_channels, stride))
        
        # 나머지 블록은 stride=1(크기 그대로)
        for _ in range(1, repeat):
            layers.append(ConvBlock(out_channels, out_channels))
        
        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)
    
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        #들어오자마자 7x7 레이어 적용함 초기 RGB(3)에서 64로 channel 옮기고, 여기서 사이즈 반으로
        #7x7 kernel, stride=2면 padding이 3이어야 정확히 반띵
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #이건 왜 3x3인지 모름, 위 블로그 참고함 아무튼 여기까지 거쳤으면
        # 초기 512x512x3 이미지가 128x128x64까지 온 상태
       
        self.layer1 = LayerBlock(64, 64, 3, 1) 
        self.layer2 = LayerBlock(64, 128, 3, 2) 
        self.layer3 = LayerBlock(128, 256, 3, 2) 
        self.layer4 = LayerBlock(256, 512, 3, 2) 
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x) # x1 : 512x512x64

        x2 = self.layer1(x1) # x2 : 512x512x64
        x3 = self.layer2(x2) # x3 : 256x256x128
        x4 = self.layer3(x3) # x4 : 128x128x256
        x5 = self.layer4(x4) # x5 : 64x64x512
        
        return x1, x2, x3, x4, x5
    

