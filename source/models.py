from torch import nn, Tensor
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split

from . import configs


class MLP(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),    
            nn.ReLU(),
            nn.Linear(512, 256),            
            nn.ReLU(),
            nn.Linear(256, 10)              
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(             # input: 3 * 32 * 32
            nn.Conv2d(3, 128, 3, padding=1),    # 128 * 32 * 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),          # 128 * 16 * 16
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 3, padding=1),  # 256 * 16 * 16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),          # 256 * 8 * 8
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 512, 3, padding=1),  # 512 * 8 * 8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),          # 512 * 4 * 4
            nn.Dropout2d(0.2),

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 10)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
class CIFAR10_Sets:
    __slots__ = ['train', 'test']

    def __init__(self):
        self.train = CIFAR10(configs.DATA_DIR, train=True, download=True)
        self.test = CIFAR10(configs.DATA_DIR, train=False, download=True)
    
    def transform(self, mean: list[float], std: list[float]) -> None:
        self.train.transform = self.test.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def train_validate_split(self, validate_size: float) -> tuple[Subset, Subset]:
        train_indices, validate_indices = train_test_split(
            range(len(self.train)),
            test_size=validate_size,
            stratify=self.train.targets,
            random_state=37
        )
        return Subset(self.train, train_indices), Subset(self.train, validate_indices)