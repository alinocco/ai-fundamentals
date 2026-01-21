from typing import Callable
import numpy as np
import math


class Conv2D:
    def __init__(
        self,
        activation_fn: Callable[[float], float],
        activation_fn_prime: Callable[[float], float],
        c_in: int,
        c_out: int,
        learning_rate = 0.01,
        k = 7,
    ):
        self.activation_fn = activation_fn
        self.activation_fn_prime = activation_fn_prime
        self.learning_rate = learning_rate
        self.c_in = c_in
        self.c_out = c_out
        self.k = k
        self.X = None
        self.z = None
        self.y = None
        std = np.sqrt(2.0 / (k * k * c_in + c_out))
        self.W = np.random.normal(0, std, size=(c_out, c_in, k, k))
        self.b = np.zeros(c_out)

    def forward_step(self, X):
        self.X = X
        h_in, w_in, c_in = X.shape
        h_out = h_in - self.k + 1
        w_out = w_in - self.k + 1
        
        self.z = np.zeros((h_out, w_out, self.c_out))
        
        for i in range(h_out):
            for j in range(w_out):
                for c_out in range(self.c_out):
                    patch = X[i:i+self.k, j:j+self.k, :]
                    self.z[i, j, c_out] = np.sum(patch * self.W[c_out].transpose(1, 2, 0)) + self.b[c_out]
        
        self.y = self.activation_fn(self.z)
        return self.y

    def backward_step(self, grad_output):
        h_in, w_in, c_in = self.X.shape
        h_out, w_out, _ = grad_output.shape
        
        grad_z = grad_output * self.activation_fn_prime(self.z)
        
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        grad_X = np.zeros_like(self.X)
        
        for i in range(h_out):
            for j in range(w_out):
                for c_out in range(self.c_out):
                    patch = self.X[i:i+self.k, j:j+self.k, :]
                    grad_W[c_out] += grad_z[i, j, c_out] * patch.transpose(2, 0, 1)
                    grad_b[c_out] += grad_z[i, j, c_out]
                    grad_X[i:i+self.k, j:j+self.k, :] += grad_z[i, j, c_out] * self.W[c_out].transpose(1, 2, 0)
        
        self.W -= self.learning_rate * grad_W
        self.b -= self.learning_rate * grad_b
        
        return grad_X