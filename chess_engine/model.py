import torch.nn as nn

class ChessBrainModel(nn.Module):
    def __init__(self):
        super(ChessBrainModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fully_connected(x)
        return x

model = ChessBrainModel()