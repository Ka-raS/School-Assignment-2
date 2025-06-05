from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(             # (_, 3, 32, 32)
            nn.Flatten(),                       # (_, 3 * 32 * 32)

            nn.Linear(3 * 32 * 32, 512),        # (_, 512)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),                # (_, 256)
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 10)                  # (_, 10)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(             # (_, 3, 32, 32)
            nn.Conv2d(3, 32, 3, padding=1),     # (_, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, stride=2),          # (_, 32, 16, 16)

            nn.Conv2d(32, 64, 3, padding=1),    # (_, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, stride=2),          # (_, 64, 8, 8)

            nn.Conv2d(64, 128, 3, padding=1),   # (_, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, stride=2),          # (_, 128, 4, 4)

            nn.Flatten(),                       # (_, 128 * 4 * 4)
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 10)          # (_, 10)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
