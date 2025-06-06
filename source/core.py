import io
import csv

import torch 
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


VALIDATION_SIZE = 0.2
EPOCH_NUM = 40
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
SGD_PARAMS = {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-3}
SCHEDULER_PARAMS = {'mode': 'min', 'factor': 0.1, 'patience': 3}


class CIFAR10Helper:
    __slots__ = (
        'train_set', 'validation_set', 'test_set', '_train_set',
        'train_loader', 'validation_loader', 'test_loader'
    )

    def __init__(self, root):
        self.train_loader = self.validation_loader = self.test_loader = None

        self._train_set = CIFAR10(root, train=True, download=True, transform=None)
        self.test_set = CIFAR10(root, train=False, download=True, transform=None)

        train_idx, validate_idx = train_test_split(
            np.arange(len(self._train_set)),
            test_size=VALIDATION_SIZE,
            stratify=self._train_set.targets,
            random_state=37,
        )
        self.train_set = Subset(self._train_set, train_idx)
        self.validation_set = Subset(self._train_set, validate_idx)
    
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
                shuffle
            )
            for dataset, batch_size, shuffle in [
                (self.train_set, BATCH_SIZE_TRAIN, True),
                (self.validation_set, BATCH_SIZE_TEST, False),
                (self.test_set, BATCH_SIZE_TEST, False)
            ]
        ]

class Trainer:
    def __init__(self, net_class: type[nn.Module], cifar10: CIFAR10Helper):
        self.cifar10 = cifar10
        self.net = net_class()
        self.criterion = nn.CrossEntropyLoss()
        if net_class.__name__ == 'MLP':
            self.optimizer = optim.Adam(self.net.parameters())
        else:
            self.optimizer = optim.SGD(self.net.parameters(), **SGD_PARAMS)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **SCHEDULER_PARAMS)

    def train(self) -> plt.Figure:
        """return learning curve plot"""

        train_losses = []
        train_accuracies = []
        validation_losses = []
        validation_accuracies = []

        epochs = range(1, EPOCH_NUM + 1)
        for epoch in epochs:
            train_loss, train_accuracy = self._train()
            validation_loss, validation_accuracy = self._test(self.cifar10.validation_loader)
            self.scheduler.step(validation_loss)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)
            
            print(f'Epoch: {epoch}/{EPOCH_NUM}, Loss: {validation_loss:.4f}, Accuracy: {validation_accuracy:.4f}', end='\r')

        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        tuples = (
            ('Loss', train_losses, validation_losses),
            ('Accuracy', train_accuracies, validation_accuracies)
        )

        for ax, (name, y_train, y_validation) in zip(axes, tuples):
            ax: plt.Axes
            ax.plot(epochs, y_train, color='blue', label='Train')
            ax.plot(epochs, y_validation, color='red', label='Validation')
            ax.set_title(f'{name} Curve')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(name)
            ax.legend()

        fig.tight_layout()
        return fig

    def test(self) -> plt.Figure:
        """"return confusion matrix plot"""

        dataset = self.cifar10.test_set
        loader = self.cifar10.test_loader
        loss, accuracy, predicts = self._test(loader, return_predictions=True)

        plt.figure(figsize=(16, 9))
        ConfusionMatrixDisplay(
            confusion_matrix(dataset.targets, predicts),
            display_labels=dataset.classes
        ).plot(ax=plt.gca(), cmap='Blues')
        
        plt.title(f'Confusion Matrix (Loss: {loss:.4f}, Accuracy: {accuracy:.4f})')
        plt.tight_layout()
        return plt.gcf()

    def _train(self) -> tuple[float, float]: 
        """return loss, accuracy"""
        
        self.net.train()
        loader = self.cifar10.train_loader
        running_loss = corrects = 0

        for images, targets in loader:
            self.optimizer.zero_grad()
            predicts = self.net(images)
            loss = self.criterion(predicts, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss
            corrects += (predicts.max(1)[1]  == targets).sum()

        loss = running_loss.item() / len(loader)
        accuracy = corrects.item() / len(loader.dataset)
        return loss, accuracy
        
    def _test(self, loader: DataLoader, return_predictions=False) -> tuple[float, float] | tuple[float, float, list[int]]: 
        """return loss, accuracy, optional predictions"""

        self.net.eval()
        running_loss = corrects = 0
        if return_predictions:
            predictions = []

        with torch.no_grad():
            for images, targets in loader:
                predicts = self.net(images)
                running_loss += self.criterion(predicts, targets)

                predicts = predicts.max(1)[1]
                corrects += (predicts == targets).sum()
                if return_predictions:
                    predictions.append(predicts)

        loss = running_loss.item() / len(loader)
        accuracy = corrects.item() / len(loader.dataset)
        if return_predictions:
            return loss, accuracy, torch.cat(predictions).tolist()
        return loss, accuracy

def plot_images_example(train_set: Subset[CIFAR10]) -> plt.Figure:
    """10x10 plot, each column is a class"""

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
            ax.set_title(train_set.dataset.classes[target])

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
    normalized_mean = mean / 255
    normalized_std = std / 255

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


# code be inherently trash and functionally garbage