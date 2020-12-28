
import time
'''
CONFIGS
'''
import torch.nn as nn

BATCH_SIZE = 128
BATCH_TEST = BATCH_SIZE
EPOCHS = 10
learning_rate = 0.01
loss_function = nn.MSELoss()
CNN_CONFIG = 'configs/c1.json'
hidden_size = 256
default_pred_file = 'pred.json'