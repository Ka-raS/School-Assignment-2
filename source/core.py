import io
import csv
from PIL.Image import Image

import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision.datasets import CIFAR10
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV


from . import configs

def plot_25_images(trainset: CIFAR10) -> plt.Figure:
    images_fig, axes = plt.subplots(5, 5, figsize=(9, 9))
    for ax, (image, label) in zip(axes.flat, trainset):
        ax: plt.Axes
        image: Image
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(trainset.classes[label])
    images_fig.tight_layout()
    return images_fig

def rgb_channels_hists(trainset: CIFAR10) -> plt.Figure:
    hists_fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharex=True, sharey=True)
    for i, ax, color in zip(range(3), axes, ['red', 'green', 'blue']):
        ax.hist(trainset.data[:, :, :, i].ravel(), bins=256, color=color)
        ax.set_title(color)
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
    hists_fig.tight_layout()
    return hists_fig

def rgb_mean_std(trainset: CIFAR10) -> tuple[list, list, io.StringIO]:
    mean = np.mean(trainset.data, axis=(0,1,2))
    std = np.std(trainset.data, axis=(0,1,2))
    normalized_mean = (mean / 256).tolist()
    normalized_std = (std / 256).tolist()

    stats_csv = io.StringIO()
    writer = csv.writer(stats_csv)
    writer.writerows([
        ['', 'Red', 'Green', 'Blue'],
        ['Mean'] + mean.tolist(),
        ['STD'] + std.tolist(),
        ['Normalized Mean'] + normalized_mean,
        ['Normalized STD'] + normalized_std
    ])
    stats_csv.seek(0)
    return normalized_mean, normalized_std, stats_csv

def grid_search_parameters(net_class: type[nn.Module], trainset: CIFAR10) -> GridSearchCV:
    param_grid = {
        'lr': configs.LEARNING_RATE,
        'batch_size': configs.BATCH_SIZE_TRAIN,
        'optimizer__momentum': configs.MOMENTUM,
        'optimizer__weight_decay': configs.WEIGHT_DECAY
    }
    net = NeuralNetClassifier(
        net_class,
        max_epochs=configs.EPOCHS // 2,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
        device=configs.DEVICE
    )
    return GridSearchCV(
        net,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=configs.CPU_CORES,
        refit=False
    ).fit(trainset.data, trainset.targets)
    
# def test():
#     ...

# def train_validate(model: nn.Module, loader: models.CIFAR10_Loader, optimizer: optim.Optimizer, criterion: nn.Module):
#     ...