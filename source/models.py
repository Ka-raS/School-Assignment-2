from torch import nn
from torch import Tensor
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split

from . import configs


class CIFAR10_Loader:
    __slots__ = ['train', 'validate', 'test']

    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(configs.CIFAR10_MEAN, configs.CIFAR10_STD)
        ])
        cifar10_train = CIFAR10(configs.DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
        train_indices, validate_indices = train_test_split(
            len(cifar10_train),
            test_size=configs.VALIDATE_SIZE,
            stratify=cifar10_train.targets,
            random_state=37
        )
        self.train = DataLoader(Subset(cifar10_train, train_indices))
        self.validate = DataLoader(Subset(cifar10_train, validate_indices))
        self.test = DataLoader(CIFAR10(configs.DATA_DIR, train=False, download=True, transform=transform))

class MultiLayerPerceptron(nn.Module): 
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

class ConvolutionalNeuralNetwork(nn.Module):
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