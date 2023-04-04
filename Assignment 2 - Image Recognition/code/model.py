import torch
import torch.nn as nn

class vggnet(nn.Module):
    def __init__(self, cfg, num_classes):
        super(vggnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.7, inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.7, inplace=False)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=num_classes, bias=True),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=-1)
        
        # initialize weights
        self.apply(self.init_weight)

    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            # He initialization with uniform distribution 
            torch.nn.init.kaiming_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.maxpool(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.softmax(x)
