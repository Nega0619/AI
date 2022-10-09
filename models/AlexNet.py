import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2) 
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2) 
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1), 
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1), 
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2) 
        )
        
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

class Alex_Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.norm1 = nn.LocalResponseNorm(size=5, k =2, alpha=0.0001, beta=0.75)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.LocalResponseNorm(size=5, k =2, alpha=0.0001, beta=0.75)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2)
                
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.flatten = nn.Flatten()

        self.drop1 = nn.Dropout(0.5)        
        self.fc1 = nn.Linear(in_features=256*6*6, out_features=2048*2)

        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=2048*2, out_features=2048*2)
        
        self.fc3 = nn.Linear(in_features=2048*2, out_features=1000)
        
    def forward(self, input_layer):
        print('input', input_layer.size())
        x = input_layer
        print(x.size())
        
        x = F.relu(self.conv1(x))
        print(x.size())
        
        x = self.norm1(x)
        print(x.size())
        
        x = self.maxpool1(x)
        print(x.size())
        
        x = F.relu(self.conv2(x))
        print(x.size())
        
        x = self.norm2(x)
        print(x.size())
        
        x = self.maxpool2(x)
        print(x.size())
        
        x = F.relu(self.conv3(x))
        print(x.size())
        
        x = F.relu(self.conv4(x))
        
        print(x.size())
        x = F.relu(self.conv5(x))
        print(x.size())
        
        x = self.maxpool5(x)
        print(x.size())

        x = self.flatten(x)
        print('flat', x.size())
        
        x = self.drop1(x)
        
        print(x.size())

        x = F.relu(self.fc1(x))
        print(x.size())
        x = self.drop2(x)

        x = F.relu(self.fc2(x))
        print(x.size())
        
        x = F.relu(self.fc3(x))
        print(x.size())

        return x

if __name__=='__main__':
    net = AlexNet()
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            

    # test = torch.randn(3,227, 227)
    summary(net, (3, 227,227))
    # net.summary()