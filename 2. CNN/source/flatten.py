from typing import Callable
import numpy as np
import math


class FlattenND:
    def __init__(self):
        self.shape = None

    def forward_pass(self, X):
        self.shape = X.shape
        return X.reshape(-1)
    
    def backward_pass(self, grad_output):
        return grad_output.reshape(self.shape)
