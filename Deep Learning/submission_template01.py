import numpy as np 
import torch 
from torch import nn

def create_model():
   
    model = nn.Sequential(
        nn.Linear(784, 256, bias=True),  # Первый линейный слой: 784 -> 256
        nn.ReLU(),                       # Функция активации ReLU
        nn.Linear(256, 16, bias=True),   # Второй линейный слой: 256 -> 16
        nn.ReLU(),                       # Функция активации ReLU
        nn.Linear(16, 10, bias=True)     # Третий линейный слой: 16 -> 10
       
    )

   
    return model

def count_parameters(model):
  
    return sum(p.numel() for p in model.parameters())
