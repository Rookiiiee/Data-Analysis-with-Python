import numpy as np
import torch
from torch import nn

def create_model():
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(784, 256, bias=True),
        nn.ReLU(),
        nn.Linear(256, 16, bias=True),
        nn.ReLU(),
        nn.Linear(16, 10, bias=True)
    )
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
    # return integer number (None is just a placeholder)
    
    return None
