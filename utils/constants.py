import math
import torch

# Training Hyperparameters: TODO: different per optimizer, find via grid search for example
EPOCHS = 10
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 1000
NUM_WORKERS = 2

LEARNING_RATE = 0.01
MOMENTUM = 0.5

TRAINING_SET_SIZE = 50000
TEST_SET_SIZE = 10000
TRAINING_BATCHES = math.ceil(TRAINING_SET_SIZE / TRAIN_BATCH_SIZE)
TEST_BATCHES = math.ceil(TEST_SET_SIZE / TEST_BATCH_SIZE)

# plot
PLOT_GRANULARITY = 5
MODEL_STORE_DIR = "./models"

# normalizing for CIFAR100
DATA_MEAN = [0.5071, 0.4867, 0.4408]
DATA_STD = [0.2675, 0.2565, 0.2761]

NUM_CLASSES = 100


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
