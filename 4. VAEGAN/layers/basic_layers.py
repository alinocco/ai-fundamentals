import numpy as cp 
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self.x = None

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, grad_out):
        pass


class AddictionalLoss(ABC):
    @abstractmethod
    def get_addictional_lose(self):
        pass


class LinearLayer(Layer):
    def __init__(self, n_in, n_out, l2_lambda=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        std = cp.sqrt(2 / (n_in + n_out))
        self.w = cp.random.normal(0, std, size=(n_in, n_out))
        self.b = cp.zeros((1, n_out), dtype=cp.float32)

        self.grad = cp.zeros_like(self.w)
        self.grad_b = cp.zeros_like(self.b)

        self.m_w = cp.zeros_like(self.w)
        self.v_w = cp.zeros_like(self.w)
        self.m_b = cp.zeros_like(self.b)
        self.v_b = cp.zeros_like(self.b)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.l2_lambda = l2_lambda

    def forward(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, grad_out):
        self.grad += self.x.T @ grad_out
        self.grad_b += cp.sum(grad_out, axis=0, keepdims=True)
        return grad_out @ self.w.T
    def reset_state(self):
        self.grad.fill(0)
        self.grad_b.fill(0)


    def update(self, lr):
        self.t += 1
        self.grad = cp.clip(self.grad, -1000, 1000)

        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * self.grad
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (self.grad ** 2)

        m_hat_w = self.m_w / (1 - self.beta1 ** self.t)
        v_hat_w = self.v_w / (1 - self.beta2 ** self.t)

        grad_w = m_hat_w + self.l2_lambda * self.w
        grad_w = cp.clip(grad_w, -1000, 1000)
        self.w -= lr * grad_w / (cp.sqrt(v_hat_w) + self.epsilon)

        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * self.grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (self.grad_b ** 2)

        m_hat_b = self.m_b / (1 - self.beta1 ** self.t)
        v_hat_b = self.v_b / (1 - self.beta2 ** self.t)

        self.b -= lr * m_hat_b / (cp.sqrt(v_hat_b) + self.epsilon)

        self.grad.fill(0)
        self.grad_b.fill(0)


class DropoutLayer(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x):
        if self.training:
            self.mask = (cp.random.rand(*x.shape) > self.p).astype(cp.float32)
            return x * self.mask / (1 - self.p)
        else:
            return x

    def backward(self, grad_out):
        if self.training and self.mask is not None:
            return grad_out * self.mask / (1 - self.p)
        else:
            return grad_out


class FlattenLayer(Layer):
    def forward(self, x):
        self.x_shape = x.shape
        N = x.shape[0]
        return x.reshape(N, -1)

    def backward(self, grad_out):
        return grad_out.reshape(self.x_shape)

