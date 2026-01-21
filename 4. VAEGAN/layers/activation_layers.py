import numpy as cp 
from .basic_layers import Layer

class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        x = cp.clip(x, -500, 500)
        self.out = cp.where(x >= 0,
                            1 / (1 + cp.exp(-x)),
                            cp.exp(x) / (1 + cp.exp(x)))
        return self.out

    def backward(self, grad_out):
        return grad_out * self.out * (1 - self.out)
class BatchNormLayer(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # Trainable parameters
        self.gamma = cp.ones((1, num_features))
        self.beta = cp.zeros((1, num_features))

        # Running stats (for inference)
        self.running_mean = cp.zeros((1, num_features))
        self.running_var = cp.ones((1, num_features))

        # Cache for backward
        self.x_centered = None
        self.std_inv = None
        self.var = None
        self.mean = None

    def forward(self, x, training=True):
        if training:
            # Batch statistics
            self.mean = cp.mean(x, axis=0, keepdims=True)
            self.var = cp.var(x, axis=0, keepdims=True)

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            # Normalize
            self.x_centered = x - self.mean
            self.std_inv = 1.0 / cp.sqrt(self.var + self.eps)
            x_norm = self.x_centered * self.std_inv
        else:
            # Inference mode
            x_norm = (x - self.running_mean) / cp.sqrt(self.running_var + self.eps)

        # Scale + shift
        out = self.gamma * x_norm + self.beta
        self.out = out
        return out

    def backward(self, grad_out):
        # Number of samples
        N = grad_out.shape[0]

        # Gradient w.r.t gamma and beta
        self.grad_gamma = cp.sum(grad_out * (self.x_centered * self.std_inv), axis=0, keepdims=True)
        self.grad_beta = cp.sum(grad_out, axis=0, keepdims=True)

        # Backprop through scale and shift
        dx_norm = grad_out * self.gamma

        # Backprop normalization
        dvar = cp.sum(dx_norm * self.x_centered * -0.5 * self.std_inv**3, axis=0, keepdims=True)
        dmean = cp.sum(dx_norm * -self.std_inv, axis=0, keepdims=True) + \
                dvar * cp.mean(-2 * self.x_centered, axis=0, keepdims=True)

        dx = dx_norm * self.std_inv + dvar * 2 * self.x_centered / N + dmean / N

        return dx

class TanhLayer(Layer):
    def forward(self, x):
        self.out = cp.tanh(x)
        return self.out

    def backward(self, grad_out):
        return grad_out * (1 - self.out**2)


class ReluLayer(Layer):
    def forward(self, x):
        self.x = x
        self.out = cp.maximum(0, x)
        return self.out

    def backward(self, grad_out):
        return grad_out * (self.x > 0)


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        x_shifted = x - cp.max(x, axis=1, keepdims=True)
        exp_x = cp.exp(x_shifted.astype(cp.float32))
        self.out = exp_x / cp.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, grad_out):
        return grad_out   


class OutputLayer(Layer):
    def forward(self, x):
        return x

    def backward(self, grad_out):
        return grad_out


