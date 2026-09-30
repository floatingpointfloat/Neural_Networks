import torch.nn as nn
import torch.nn.functional as F

class Resblock(nn.Module):
    def __init__(self, channels):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + residual
        out = F.relu(out)

        return out
    
class ChessValueNet(nn.Module):
    def __init__(self):
        super(ChessValueNet, self).__init__()

        
        self.start_conv = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.features = nn.Sequential(
            Resblock(128),
            Resblock(128),
            Resblock(128),
            Resblock(128),
            Resblock(128)
        )

        self.fully_connected = nn.Sequential(
            nn.Flatten(), #fc layers expect a vector, not feature maps
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            #nn.Dropout(0.3), #prevent overfitting
            nn.Linear(256, 1),
            nn.Tanh() #normalize betwenn -1 and 1
        )

    def forward(self, x):
        x = self.start_conv(x)
        x = self.features(x)
        x = self.fully_connected(x)

        return x