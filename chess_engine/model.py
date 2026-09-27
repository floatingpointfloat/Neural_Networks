import torch.nn as nn

class ChessValueNet(nn.Module):
    def __init__(self):
        super(ChessValueNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.ReLU(), #making the model nonlinear - better :) can learn complex stuff
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fully_connected = nn.Sequential(
            nn.Flatten(), #fc layers expect a vector, not feature maps
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3), #prevent overfitting
            nn.Linear(256, 1),
            nn.Tanh() #normalize betwenn -1 and 1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fully_connected(x)
        return x