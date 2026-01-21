from typing import Callable
import numpy as np
import math

class MaxPooling2D:
    def __init__(self, size=2):
        self.size = size
        self.X = None
        self.y = None
        self.positions = {}

    def forward_pass(self, X):
        self.X = X
        self.y = np.zeros((X.shape[0] // self.size, X.shape[1] // self.size, X.shape[2]))
        for i in range(X.shape[0] // self.size):
            for j in range(X.shape[1] // self.size):
                for k in range(X.shape[2]):
                    target_slice = X[self.size * i: self.size * (i + 1), self.size * j: self.size * (j + 1), k]
                    self.y[i][j][k] = np.max(target_slice)
                    self.positions[(i, j, k)] = np.unravel_index(np.argmax(target_slice, axis=None), target_slice.shape)
        return self.y

    def backward_pass(self, grad_output):
        target_grad = np.zeros(self.X.shape)
        for i in range(grad_output.shape[0]):
            for j in range(grad_output.shape[1]):
                for k in range(grad_output.shape[2]):
                    local_pos = self.positions[(i, j, k)]
                    local_row = int(local_pos[0])
                    local_col = int(local_pos[1])
                    global_row = local_row + self.size * i
                    global_col = local_col + self.size * j
                    target_grad[global_row, global_col, k] = grad_output[i][j][k]
        return target_grad
    