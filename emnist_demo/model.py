import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x  # residual point :-)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)

        return out


class CNN_ResNet_Model(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # stage 1: 32 channels
        self.resblock1 = ResBlock(32)
        self.resblock2 = ResBlock(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # stage 2: 64 channels
        self.transition1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.resblock3 = ResBlock(64)
        self.resblock4 = ResBlock(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # stage 3: 128 channels
        self.transition2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.resblock5 = ResBlock(128)

        # classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.initial(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.pool1(x)

        x = self.transition1(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.pool2(x)

        x = self.transition2(x)
        x = self.resblock5(x)

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)

        return x
