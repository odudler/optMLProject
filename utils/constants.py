import math
import torch

#number of epochs in training
EPOCHS = 10

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 1000
#number of workers for parallel training
NUM_WORKERS = 2

#hyper parameters for debugging purposes only
LEARNING_RATE = 0.01
MOMENTUM = 0.5

#dataset specific parameters (CIFAR-100)
TRAINING_SET_SIZE = 50000
TEST_SET_SIZE = 10000
TRAINING_BATCHES = math.ceil(TRAINING_SET_SIZE / TRAIN_BATCH_SIZE)
TEST_BATCHES = math.ceil(TEST_SET_SIZE / TEST_BATCH_SIZE)
# normalizing for CIFAR100
DATA_MEAN = [0.5071, 0.4867, 0.4408]
DATA_STD = [0.2675, 0.2565, 0.2761]
NUM_CLASSES = 100

#constants for plotting
PLOT_GRANULARITY = 5
MODEL_STORE_DIR = "./models"

#check if gpu available, store in DEVICE constant
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
