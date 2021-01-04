
import time
'''
CONFIGS
'''
import torch.nn as nn

BATCH_SIZE = 64
BATCH_TEST = BATCH_SIZE
EPOCHS = 30
learning_rate = 0.001
CNN_CONFIG = 'configs/c1.json'
hidden_size = 256
word_embeddings_size = 256
slot_embeddings_size = 256
default_pred_file = 'pred.json'
lstm_dr = 0.0
lstm_bidirectional = False