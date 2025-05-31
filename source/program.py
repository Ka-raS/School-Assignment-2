import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

# from . import configs, core, models
import configs, core, models


def prepare() -> None:
    configs.DATA_DIR.mkdir(exist_ok=True)
    configs.OUTPUT_DIR.mkdir(exist_ok=True)

    random.seed(configs.SEED)
    np.random.seed(configs.SEED)
    torch.manual_seed(configs.SEED)

def data_analysis(train_set: Subset[CIFAR10]) -> tuple[np.ndarray, np.ndarray]:
    images = core.plot_images_example(train_set)
    images_dir = configs.OUTPUT_DIR / 'cifar10-images.pdf'
    images.savefig(images_dir)
    print(images_dir)

    rgb_hists = core.rgb_channels_hists(train_set)
    rgb_hists_dir = configs.OUTPUT_DIR / 'rgb-hists.pdf'
    rgb_hists.savefig(rgb_hists_dir)
    print(rgb_hists_dir)
    
    mean, std, stats_csv = core.rgb_mean_std(train_set)
    stats_csv_dir = configs.OUTPUT_DIR / 'rgb-mean-std.csv'
    with open(stats_csv_dir, 'w', newline='') as file:
        file.write(stats_csv.read())
    print(stats_csv_dir)

    plt.close('all')
    return mean, std

def train_validate_test(net_class: type[nn.Module], cifar10: core.CIFAR10Helper) -> None:
    name = net_class.__name__.lower()
    trainer = core.Trainer(net_class, cifar10)

    learning_curve_plot = trainer.train()
    learning_curve_dir = configs.OUTPUT_DIR / f'{name}-learning-curve.pdf'
    learning_curve_plot.savefig(learning_curve_dir)
    print(learning_curve_dir)

    test_result_plot = trainer.test()
    test_result_dir = configs.OUTPUT_DIR / f'{name}-test-result.pdf'
    test_result_plot.savefig(test_result_dir)
    print(test_result_dir)

    model_dir = configs.OUTPUT_DIR / f'{name}-model.pth'
    torch.save(trainer.net, model_dir)
    print(model_dir)

    plt.close('all')

def run() -> None:
    prepare()

    print('Data Analysis:')
    cifar10 = core.CIFAR10Helper()
    mean, std = data_analysis(cifar10.train_set)
    cifar10.normalize(mean, std)
 
    print('\nTrain / Validate / Test:')
    cifar10.make_loaders()
    train_validate_test(models.MLP, cifar10)
    train_validate_test(models.CNN, cifar10)
