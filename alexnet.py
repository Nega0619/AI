from doctest import OutputChecker
from re import X
from pandas import infer_freq
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005



class AlexNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        # relu
        self.norm1 = nn.LocalResponseNorm(size=5, k =2, alpha=0.0001, beta=0.75)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5)
        # relu
        self.norm2 = nn.LocalResponseNorm(size=5, k =2, alpha=0.0001, beta=0.75)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2)
                
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3)
        # relu
        
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3)
        # relu
        
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3)
        # relu
        self.maxpool5 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.flatten = nn.Flatten()
        
        self.drop = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(in_features=2048*2, out_features=2048*2)
        # relu
        
        self.fc2 = nn.Linear(in_features=2048*2, out_features=2048*2)
        # relu
        
        self.fc3 = nn.Linear(in_features=2048*2, out_features=1000)
        
    def forward(self, input_layer):
        x = input_layer
        
        x = F.relu(self.conv1(x))
        x = self.norm1(x)
        x = self.maxpool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.maxpool2(x)
        
        x = F.relu(self.conv3(x))
        
        x = F.relu(self.conv4(x))
        
        x = F.relu(self.conv5(x))
        x = self.maxpool5(x)
        
        x = self.flatten(x)
        
        x = self.drop(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

net = AlexNet()        
        
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)