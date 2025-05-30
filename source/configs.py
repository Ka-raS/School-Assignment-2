import pathlib
import multiprocessing

import torch


ROOT_DIR = pathlib.Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / 'data'
DATA_ANALYSIS_DIR = ROOT_DIR / 'output/data-analysis'
PARAMS_TUNING_DIR = ROOT_DIR / 'output/params-tuning'
TRAIN_TEST_DIR = ROOT_DIR / 'output/train-test'

SEED = 37
CPU_CORES = multiprocessing.cpu_count() - 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOG_INTERVAL_TRAIN = 20
VALIDATE_SIZE = 0.2
EPOCHS = 10
BATCH_SIZE_TEST = 1000
BATCH_SIZE_TRAIN = [32, 64, 128]
LEARNING_RATE = [0.001, 0.001, 0.1]
MOMENTUM = [0.9, 0.95]
WEIGHT_DECAY = [0.0001, 0.0005]
