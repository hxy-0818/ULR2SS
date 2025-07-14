import os
import torch

LOAD_MODEL = False
SAVE_MODEL = True
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 16
LAMBDA_GP = 10
LOW_RES = 96
HIGH_RES = LOW_RES * 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INTERMEDIA_SIZE = 16

IMG_CHANNELS = 3

