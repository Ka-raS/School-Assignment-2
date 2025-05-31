import pathlib
import multiprocessing

import torch


ROOT_DIR = pathlib.Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / 'data'
OUTPUT_DIR = ROOT_DIR / 'output'

SEED = 37
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU_CORES = multiprocessing.cpu_count() - 1

VALIDATION_SIZE = 0.2  
EPOCHS = 15
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LOG_INTERVAL = 20

LEARNING_RATE = 0.01
MOMENTUM = 0.95
WEIGHT_DECAY = 1e-4