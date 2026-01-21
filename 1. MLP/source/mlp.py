from typing import List, Callable
from tqdm import tqdm  # type: ignore
import random 
import numpy as np

class MLP:
  def __init__(
                self,
                x_size: int,
                y_size: int,
                h_sizes: List[int],
                activation_fn: Callable[[float], float],
                activation_fn_prime: Callable[[float], float],
                learning_rate: float
        ):
    self.x = np.zeros(x_size)
    self.y = np.zeros(y_size)
    self.z = [0] * (len(h_sizes) + 1)
    self.a = [0] * (len(h_sizes) + 1)
    self.w = []
    self.b = []
    self.activation_fn = activation_fn
    self.activation_fn_prime = activation_fn_prime
    self.learning_rate = learning_rate

    # init weight matrix, but not values of weights
    self.w.append(np.zeros((h_sizes[0], x_size)))
    self.b.append(np.zeros(h_sizes[0]))
    for i in range(len(h_sizes) - 1):
      self.w.append(np.zeros((h_sizes[i + 1], h_sizes[i])))
      self.b.append(np.zeros(h_sizes[i + 1]))
    self.w.append(np.zeros((y_size, h_sizes[-1])))
    self.b.append(np.zeros(y_size))

    # init weights values
    for i in range(len(self.w)):
      self.w[i][:] = np.random.normal(0, np.sqrt(2.0 / self.w[i].shape[1]), size=self.w[i].shape)
      self.b[i][:] = np.zeros(self.b[i].shape)

  def forward_step(self):
    self.z[0] = self.w[0] @ self.x + self.b[0]
    self.a[0] = self.activation_fn(self.z[0])
    for i in range(1, len(self.w) - 1):
      self.z[i] = self.w[i] @ self.a[i - 1] + self.b[i]
      self.a[i] = self.activation_fn(self.z[i])
    self.z[-1] = self.w[-1] @ self.a[-2] + self.b[-1]
    self.a[-1] = self.z[-1]
    self.y = self.a[-1]

  def backward_step(self, y_actual: List[float], external_grad=None):
    if external_grad is not None:
        grad = external_grad
    else:
        grad = 2 * (self.y - y_actual)
    
    grad_input = None
    for i in range(len(self.w) - 1, -1, -1):
        if i == len(self.w) - 1 and external_grad is not None:
            grad_z = grad
        else:
            grad_z = grad * self.activation_fn_prime(self.z[i])
        
        if i == 0:
            layer_input = self.x
            grad_input = self.w[i].T @ grad_z
        else:
            layer_input = self.a[i - 1]  # type: ignore
            grad = self.w[i].T @ grad_z
        
        self.w[i] -= self.learning_rate * np.outer(grad_z, layer_input)
        self.b[i] -= self.learning_rate * grad_z
    return grad_input

  def fit(self, X, Y, epochs=1):
    for epoch in tqdm(range(epochs)):
      _xy = list(zip(X, Y))
      random.shuffle(_xy)
      for x, y in _xy:
        self.x = x
        self.forward_step()
        self.backward_step(y)

  def predict(self, x):
    self.x = x
    self.forward_step()
    return self.y