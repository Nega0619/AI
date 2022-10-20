import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchsummary import summary

class BasicConv2d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x:Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)            # inplace = True 옵션생략했음.

class inception_block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3_in: int,
        ch3x3: int,
        ch5x5_in: int,
        ch5x5: int,
        pool_proj: int,
        ) -> None:
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1, stride=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3_in, kernel_size = 1, stride=1),
            BasicConv2d(ch3x3_in, ch3x3, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5_in, kernel_size = 1, stride=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            BasicConv2d(ch5x5_in, ch5x5, kernel_size=3, stride=1, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x:Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # print(branch1.size())
        # print(branch2.size())
        # print(branch3.size())
        # print(branch4.size())

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class my_GoogLeNet(nn.Module):
    def __init__(self, input_size=224):
        # super.__init__()
        super(my_GoogLeNet, self).__init__()

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)     # padding = 3인데 일단 적용안하고 해봄 (안하면 output size가 맞지않음..)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)               # ceil_mode=True 인데 이거안해주면 output size 차이남.
        self.LocalRespNorm1 = nn.LocalResponseNorm(64)                          # pytorch 모델에는 없지만, 논문에는 있어서 넣어봄.
                                                                                # https://pytorch.org/docs/stable/generated/torch.nn.LocalResponseNorm.html
        self.conv2 = BasicConv2d(64, 64, kernel_size=1, stride = 1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, stride = 1, padding=1) # padding = 1안하면 output안맞음..
        self.LocalRespNorm2 = nn.LocalResponseNorm(192)
        
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)        

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128) 

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(1024, 1000)               

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.LocalRespNorm1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.LocalRespNorm2(x)

        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)

        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

test = torch.randn(224,224, 3)
net = my_GoogLeNet()
summary(net, (3, 224, 224))        