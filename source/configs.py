import multiprocessing
import pathlib

import torch


ROOT_DIR = pathlib.Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / 'data'
OUTPUT_DIR = ROOT_DIR / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

VALIDATE_SIZE = 10_000 / 50_000
CIFAR10_NORM_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR10_NORM_STD = (0.24703233, 0.24348505, 0.26158768)

EPOCHS = 10
LOG_INTERVAL = 10

# LEARNING_RATE = 0.001
# BETAS = (0.9, 0.999)
# EPSILON = 1e-8
# WEIGHT_DECAY = 0.01

NUM_WORKERS = multiprocessing.cpu_count() - 1
torch.set_num_threads(torch.get_num_threads() - 1)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
