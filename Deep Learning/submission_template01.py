import numpy as np
import torch
from torch import nn

def create_model():
    model = nn.Sequential(
        nn.Linear(784, 256, bias=True),
        nn.ReLU(),
        nn.Linear(256, 16, bias=True),
        nn.ReLU(),
        nn.Linear(16, 10, bias=True)
    )
    # return model instance (None is just a placeholder)
    return model

def count_parameters(model):
    # your code here
    # return integer number (None is just a placeholder)
    return sum(p.numel() for p in model.parameters())
    return None
