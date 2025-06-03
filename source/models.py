from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        relu = nn.ReLU()
        dropout = nn.Dropout(0.2)

        self.__model = nn.Sequential(           # (_, 3, 32, 32)
            nn.Flatten(),                       # (_, 3 * 32 * 32)

            nn.Linear(3 * 32 * 32, 512),        # (_, 512)
            nn.BatchNorm1d(512),
            relu,
            dropout,

            nn.Linear(512, 256),                # (_, 256)
            nn.BatchNorm1d(256),
            relu,
            dropout,

            nn.Linear(256, 10)                  # (_, 10)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.__model(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        relu = nn.ReLU()
        maxpool = nn.MaxPool2d(2, stride=2)
        dropout = nn.Dropout2d(0.2)

        self._model = nn.Sequential(            # (_, 3, 32, 32)
            nn.Conv2d(3, 128, 3, padding=1),    # (_, 128, 32, 32)
            nn.BatchNorm2d(128),
            relu,
            maxpool,                            # (_, 128, 16, 16)
            dropout,

            nn.Conv2d(128, 256, 3, padding=1),  # (_, 256, 16, 16)
            nn.BatchNorm2d(256),
            relu,
            maxpool,                            # (_, 256, 8, 8)
            dropout,

            nn.Conv2d(256, 512, 3, padding=1),  # (_, 512, 8, 8)
            nn.BatchNorm2d(512),
            relu,
            maxpool,                            # (_, 512, 4, 4)
            dropout,

            nn.Flatten(),                       # (_, 512 * 4 * 4)
            nn.Linear(512 * 4 * 4, 10)          # (_, 10)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)
