import random
import pathlib

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

from . import core, models


ROOT_DIR = pathlib.Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / 'data'
OUTPUT_DIR = ROOT_DIR / 'output'


def prepare() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    random.seed(37)
    np.random.seed(37)
    torch.manual_seed(37)

def data_analysis(train_set: Subset[CIFAR10]) -> tuple[np.ndarray, np.ndarray]:
    images = core.plot_images_example(train_set)
    images_dir = OUTPUT_DIR / 'cifar10-images.pdf'
    images.savefig(images_dir)
    print(images_dir)

    channels_hists = core.rgb_channels_hists(train_set)
    channels_hists_dir = OUTPUT_DIR / 'cifar10-channels.pdf'
    channels_hists.savefig(channels_hists_dir)
    print(channels_hists_dir)
    
    mean, std, file_csv = core.rgb_mean_std(train_set)
    file_csv_dir = OUTPUT_DIR / 'cifar10-mean-std.csv'
    with open(file_csv_dir, 'w', newline='') as file:
        file.write(file_csv.read())
    print(file_csv_dir)

    plt.close('all')
    return mean, std

def train_validate_test(net_class: type[nn.Module], cifar10: core.CIFAR10Helper) -> None:
    name = net_class.__name__
    print(name + ':')
    name = name.lower()
    trainer = core.Trainer(net_class, cifar10)

    learning_curve = trainer.train()
    learning_curve_dir = OUTPUT_DIR / f'{name}-learning-curve.pdf'
    learning_curve.savefig(learning_curve_dir)
    print(learning_curve_dir)

    confusion_matrix = trainer.test()
    confusion_matrix_dir = OUTPUT_DIR / f'{name}-confusion-matrix.pdf'
    confusion_matrix.savefig(confusion_matrix_dir)
    print(confusion_matrix_dir)

    model_dir = OUTPUT_DIR / f'{name}-model.pth'
    torch.save(trainer.net, model_dir)
    print(model_dir)

    plt.close('all')

def run2() -> None:
    prepare()

    print('Data Analysis:')
    cifar10 = core.CIFAR10Helper(DATA_DIR)
    mean, std = data_analysis(cifar10.train_set)
    cifar10.normalize(mean, std)
 
    print('\nTrain / Validate / Test:')
    cifar10.make_loaders()
    train_validate_test(models.MLP, cifar10)
    train_validate_test(models.CNN, cifar10)
