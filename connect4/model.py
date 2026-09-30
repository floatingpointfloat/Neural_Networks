"""
Model based on the alphazero-connect4 model by steven0226 (weights and config on huggingface)
-> is the same shape as his, so the parameters of his model fit this one (he used it in combinatino with a policy head though)
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, filters: int):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            filters, filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(
            filters, filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        return torch.relu(x + y)


class ValueNet(nn.Module):
    def __init__(self, blocks=6, filters=96):
        super(ValueNet, self).__init__()

        self.blocks = blocks
        self.filters = filters

        self.stem = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )

        self.tower = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])

        flat = 32 * 6 * 7

        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.tower(h)
        value = self.value_head(h)

        return value.squeeze(-1)
