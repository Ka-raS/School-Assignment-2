import io
import csv

import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split

# from . import configs
import configs


class CIFAR10Helper:
    __slots__ = (
        'train_set', 'validation_set', 'test_set', '_train_set'
        'train_loader', 'validation_loader', 'test_loader'    
    )

    def __init__(self):
        self._train_set = CIFAR10(configs.DATA_DIR, train=True, download=True, transform=None)
        self.test_set = CIFAR10(configs.DATA_DIR, train=False, download=True, transform=None)

        train_idx, validate_idx = train_test_split(
            np.arange(len(self._train_set)),
            test_size=configs.VALIDATION_SIZE,
            stratify=self._train_set.targets,
            random_state=configs.SEED,
        )
        self.train_set = Subset(self._train_set, train_idx)
        self.validation_set = Subset(self._train_set, validate_idx)

        self.train_loader, self.validation_loader, self.test_loader = None
    
    def normalize(self, mean, std) -> None:
        # pytorch lazy implement 
        self._train_set.transform = self.test_set.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def make_loaders(self) -> None:
        self.train_loader, self.validation_loader, self.test_loader = [
            DataLoader(
                dataset, 
                batch_size, 
                shuffle, 
                num_workers=configs.CPU_CORES, 
                pin_memory_device=configs.DEVICE
            )
            for dataset, batch_size, shuffle in [
                (self.train_set, configs.BATCH_SIZE_TRAIN, True),
                (self.validation_set, configs.BATCH_SIZE_TEST, False),
                (self.test_set, configs.BATCH_SIZE_TEST, False)
            ]
        ]

class Trainer:
    def __init__(self, net_class: type[nn.Module], cifar10: CIFAR10Helper):
        self.net = net_class().to(configs.DEVICE)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=configs.LEARNING_RATE,
            momentum=configs.MOMENTUM,
            weight_decay=configs.WEIGHT_DECAY
        )
        self.cifar10 = cifar10

    def train(self) -> plt.Figure:
        """return learning curve plot"""

        train_losses = []
        train_accuracies = []
        validation_losses = []
        validation_accuracies = []

        for epoch in range(configs.EPOCHS):
            train_loss, train_accuracy = self._train(self.cifar10.train_loader)
            validation_loss, validation_accuracy = self._test(self.cifar10.validation_loader)

            train_losses.extend(train_loss)
            train_accuracies.extend(train_accuracy)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)

        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        values = [
            ('Loss', train_losses, validation_losses),
            ('Accuracy', train_accuracies, validation_accuracies)
        ]

        for ax, (name, trains, validations) in zip(axes, values):
            ax: plt.Axes
            ax.plot(trains, color='blue', label='Train')
            ax.plot(validations, color='red', label='Validation')
            ax.set_title(f'{name} Curve')
            ax.set_xlabel('Batches Trained')
            ax.set_ylabel(name)

        fig.tight_layout()
        return fig

    def test(self) -> plt.Figure:
        """"plot: loss, accuracy, confusion matrix"""
        pass

    def _train(self, loader) -> tuple[list[float], list[float]]: 
        """loss, accuracy per configs.LOG_INTERVAL"""
        self.net.train()
        
    def _test(self, loader) -> tuple[float, float]: 
        """loss, accuracy"""
        self.net.eval()

def plot_images_example(train_set: Subset[CIFAR10]) -> plt.Figure:
    """10x10 plot, each column is a class"""

    classes = train_set.dataset.classes
    rows = [0] * 10
    columns_full = 0
    fig, axes = plt.subplots(10, 10, figsize=(9, 9))

    for image, target in train_set:
        if rows[target] >= 10:
            continue

        ax: plt.Axes = axes[rows[target], target]
        ax.imshow(image)
        ax.axis('off')
        if rows[target] == 0:
            ax.set_title(classes[target])

        rows[target] += 1
        if rows[target] == 10:
            columns_full += 1
            if columns_full == 10:
                break

    fig.tight_layout()
    return fig

def rgb_channels_hists(train_set: Subset[CIFAR10]) -> plt.Figure:
    """raw rgb channels"""

    data = train_set.dataset.data[train_set.indices]
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharex=True, sharey=True)

    for i, ax, color in zip(range(3), axes, ['red', 'green', 'blue']):
        ax: plt.Axes
        ax.hist(data[:, :, :, i].ravel(), bins=256, color=color)
        ax.set_title(color)
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')

    fig.tight_layout()
    return fig

def rgb_mean_std(train_set: Subset[CIFAR10]) -> tuple[np.ndarray, np.ndarray, io.StringIO]:
    """return mean, std, csv"""

    data = train_set.dataset.data[train_set.indices]    # (40_000, 32, 32, 3)
    mean = np.mean(data, axis=(0,1,2))
    std = np.std(data, axis=(0,1,2))
    normalized_mean = mean / 256
    normalized_std = std / 256

    csv_file = io.StringIO(newline='')
    writer = csv.writer(csv_file)
    writer.writerows([
        ['', 'Red', 'Green', 'Blue'],
        ['Mean'] + mean.tolist(),
        ['STD'] + std.tolist(),
        ['Normalized Mean'] + normalized_mean.tolist(),
        ['Normalized STD'] + normalized_std.tolist()
    ])

    csv_file.seek(0)
    return normalized_mean, normalized_std, csv_file
