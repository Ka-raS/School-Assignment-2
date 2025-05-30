import csv
import random

import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10

from . import (
    configs,
    core,
    models
)


def prepare() -> None:
    configs.DATA_DIR.mkdir(parents=True, exist_ok=True)
    configs.DATA_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    configs.PARAMS_TUNING_DIR.mkdir(parents=True, exist_ok=True)
    configs.TRAIN_TEST_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(configs.SEED)
    np.random.seed(configs.SEED)
    torch.manual_seed(configs.SEED)

def data_analysis(trainset: CIFAR10) -> tuple[list, list]:
    images = core.plot_25_images(trainset)
    hists = core.rgb_channels_hists(trainset)
    normalized_mean, normalized_std, stats_csv = core.rgb_mean_std(trainset)
    
    images_dir = configs.DATA_ANALYSIS_DIR / 'cifar10-images.pdf'
    hists_dir = configs.DATA_ANALYSIS_DIR / 'rgb-hists.pdf'
    stats_csv_dir = configs.DATA_ANALYSIS_DIR / 'rgb-mean-std.csv'

    images.savefig(images_dir)
    hists.savefig(hists_dir)
    with open(stats_csv_dir, 'w', newline='') as file:
        file.write(stats_csv.read())

    print(images_dir)
    print(hists_dir)
    print(stats_csv_dir)

    return normalized_mean, normalized_std

def hyper_parameters_tuning(cls: type[nn.Module], trainset: CIFAR10) -> dict[str, int | float]:
    gs = core.grid_search_parameters(cls, trainset)
    csv_dir = configs.PARAMS_TUNING_DIR / f'{cls.__name__}-grid-search.csv'

    with open(csv_dir, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(gs.cv_results_.keys())
        writer.writerows(np.array(list(gs.cv_results_.values())).T)

    print(csv_dir)
    return gs.best_params_

def train_test_model(cls: type[nn.Module], params: dict[str, int | float], cifar10: models.CIFAR10_Sets) -> None:
    

def run() -> None:
    prepare()

    print('Data Analysis:')
    cifar10 = models.CIFAR10_Sets()
    normalized_mean, normalized_std = data_analysis(cifar10.train)
    cifar10.transform(normalized_mean, normalized_std)

    print('\Hyper Parameters Tuning:')
    mlp_params = hyper_parameters_tuning(models.MLP, cifar10.train)
    cnn_params = hyper_parameters_tuning(models.CNN, cifar10.train)

    print('\nTrain And Test Models:')
    train_test_model(models.MLP, mlp_params, cifar10)
    train_test_model(models.CNN, cnn_params, cifar10)