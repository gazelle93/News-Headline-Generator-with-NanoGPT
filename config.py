import os
import os.path as op
import torch
import torch.nn as nn

class BaseConfig(object):
    BASEDIR = op.abspath(op.dirname(__file__))
    PROJECT_ROOT = BASEDIR
    # set device
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    VOCAB_SIZE = 5_000
    MIN_FREQUENCY = 2


    vocab_size = VOCAB_SIZE
    block_size = 256
    emb_dim = 512
    num_heads = 4
    num_layers = 1
    dropout = 0.1
    dim_expansion = 4
    bias = False

    initial_lr = 3e-4
    min_lr = 1e-4

    batch_size = 450

Config = BaseConfig