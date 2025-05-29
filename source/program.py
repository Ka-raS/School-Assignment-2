from torch import nn, optim

from . import (
    configs,
    core,
    models
)


def data_analysis() -> None:
    images, hists, stats_csv = core.analize_CIFAR10_intensity()
    
    images_dir = configs.OUTPUT_DIR / 'cifar10-images.pdf'
    hists_dir = configs.OUTPUT_DIR / 'rgb-intensity-hists.pdf'
    stats_csv_dir = configs.OUTPUT_DIR / 'intensity-mean-std.csv'

    images.savefig(images_dir)
    hists.savefig(hists_dir)
    with open(stats_csv_dir, 'w') as file:
        file.write(stats_csv.read())

    print(images_dir)
    print(hists_dir)
    print(stats_csv_dir)

def run() -> None:
    data_analysis()

    loader = models.CIFAR10_Loader()
    mlp_net = models.MultiLayerPerceptron().to(configs.DEVICE)
    cnn_net = models.ConvolutionalNeuralNetwork().to(configs.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW()

    mlp_learning_curves = core.train_validate_plot(mlp, loader, criterion, optimizer)
    cnn_learning_curves = core.train_validate_plot(mlp, loader, criterion, optimizer)