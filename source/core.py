import io
import csv
from PIL.Image import Image

import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import CIFAR10

from . import (
    configs, 
    models
)


def analize_CIFAR10_intensity() -> tuple[plt.Figure, plt.Figure, io.StringIO]:
    """25 images rgb intensity hists, mean median std csv"""
    
    dataset = CIFAR10(configs.DATA_DIR, download=True)
    intensities = np.arange(256)
    rgb_frequencies = [[] for _ in range(3)]
    images_fig, axes = plt.subplots(5, 5, figsize=(9, 9))

    for ax, (image, label) in zip(axes.flat, dataset):
        ax: plt.Axes
        image: Image
        ax.imshow(image)
        ax.set_title(dataset.classes[label])
        ax.axis('off')
        for frequencies, color_image in zip(rgb_frequencies, image.split()):
            frequencies.append(color_image.histogram())

    hists_fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharex=True, sharey=True)
    rgb_mean_frequencies = [np.mean(frequencies, axis=0) for frequencies in rgb_frequencies]

    for ax, color, frequencies in zip(axes, ['red', 'green', 'blue'], rgb_mean_frequencies):
        ax.plot(intensities, frequencies, color=color)
        ax.set_title(color)
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')

    grid = [
        ['Color', 'Mean', 'Standard Deviation'],
        ['Red'],
        ['Green'],
        ['Blue']
    ]
    for i, frequencies in enumerate(rgb_mean_frequencies, start=1):
        mean = np.average(intensities, weights=frequencies),
        std = np.sqrt(np.average((intensities - mean) ** 2, weights=frequencies))
        grid[i].extend([mean, std])

    stats_csv = io.StringIO(newline='')
    writer = csv.writer(stats_csv)
    writer.writerows(grid)

    images_fig.tight_layout()
    hists_fig.tight_layout()
    return images_fig, hists_fig, stats_csv


def test():
    ...

def train_validate(model: nn.Module, loader: models.CIFAR10_Loader, optimizer: optim.Optimizer, criterion: nn.Module):
    ...