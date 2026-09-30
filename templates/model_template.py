import torch.nn as nn
import torch.nn.functional as F

#simplevalue, cnn, cnnresnet

class SimpleValueNet(nn.Module):
    def __init__(
        self,
        input_channels,
        board_height,
        board_width,
        hidden_size=256
    ):
        super(SimpleValueNet, self).__init__()

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                input_channels * board_height * board_width,
                hidden_size
            ),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fully_connected(x)

        return x


class CNNValueNet(nn.Module):
    def __init__(
        self,
        input_channels,
        board_height,
        board_width,
        channels=128,
        hidden_size=256,
        dropout=0.3
    ):
        super(CNNValueNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                channels * board_height * board_width,
                hidden_size
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fully_connected(x)

        return x


class Resblock(nn.Module):
    def __init__(self, channels):
        super(Resblock, self).__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + residual
        out = F.relu(out)

        return out


class CNNResNetValueNet(nn.Module):
    def __init__(
        self,
        input_channels,
        board_height,
        board_width,
        channels=128,
        hidden_size=256,
        resblocks=5,
        dropout=0.3
    ):
        super(CNNResNetValueNet, self).__init__()

        self.start_conv = nn.Sequential(
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                64,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.features = nn.Sequential(
            *[
                Resblock(channels)
                for _ in range(resblocks)
            ]
        )

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                channels * board_height * board_width,
                hidden_size
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.start_conv(x)
        x = self.features(x)
        x = self.fully_connected(x)

        return x