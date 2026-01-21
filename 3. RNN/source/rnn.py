import numpy as np

class RNN():

    def _init_weights(self):
        self.w1 = np.random.normal(0, np.sqrt(2.0 / self.w1.shape[1]), size=self.w1.shape)
        self.w2 = np.random.normal(0, np.sqrt(2.0 / self.w2.shape[1]), size=self.w2.shape)
        self.w3 = np.random.normal(0, np.sqrt(2.0 / self.w3.shape[1]), size=self.w3.shape)
        self.b1 = np.zeros(self.hidden_size)
        self.b2 = np.zeros(self.output_size)

    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w1 = np.zeros((hidden_size, hidden_size))
        self.w2 = np.zeros((input_size, hidden_size))
        self.w3 = np.zeros((hidden_size, output_size))
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(output_size)
        self.lr = lr
        self._init_weights()

    def load_weights_from_RNN(self, rnn):
        self.w1 = rnn.w1.weight.data.numpy()
        self.w2 = rnn.w2.weight.data.numpy()
        self.w3 = rnn.w3.weight.data.numpy()
        if hasattr(rnn.w1, 'bias') and rnn.w1.bias is not None:
            self.b1 = rnn.w1.bias.data.numpy()
        if hasattr(rnn.w3, 'bias') and rnn.w3.bias is not None:
            self.b2 = rnn.w3.bias.data.numpy()

    def forward(self, x: np.ndarray, hx = None):
        h_history = []
        z_history = []

        assert x.shape[1] == self.input_size, "Input size mismatch"
        seq_size, _ = x.shape
        if hx is None:
            hx = np.zeros((self.hidden_size))
        h_t = hx.copy()
        outputs = []
        for i in range(seq_size):
            z_t = self.w1 @ h_t + self.w2.T @ x[i,:] + self.b1
            h_t = np.tanh(z_t)
            
            z_history.append(z_t)
            h_history.append(h_t)
            outputs.append(self.w3.T @ h_t + self.b2)
        
        self.z_history = np.array(z_history)
        self.h_history = np.array(h_history)

        return np.array(outputs), h_t
        
    def backward(self, x: np.ndarray, y: np.ndarray):
        seq_size, _ = x.shape
        _, output_size = y.shape
        outputs = self.forward(x)[0].reshape(-1, output_size)
        loss = np.mean((y - outputs) ** 2)

        grad_w1 = np.zeros_like(self.w1)
        grad_w2 = np.zeros_like(self.w2)
        grad_w3 = np.zeros_like(self.w3)
        grad_b1 = np.zeros_like(self.b1)
        grad_b2 = np.zeros_like(self.b2)

        dh_next = np.zeros((self.hidden_size, 1))

        for i in reversed(range(seq_size)):
            dl_dy = 2 * (outputs[i] - y[i]).reshape(-1, 1)
            dy_dw3 = self.h_history[i].reshape(-1, 1)

            if i > 0:
                h_prev = self.h_history[i-1].reshape(-1, 1)
            else:
                h_prev = np.zeros((self.hidden_size, 1))

            dl_dh = self.w3 @ dl_dy + dh_next
            dh_dz = (1 - np.tanh(self.z_history[i])**2).reshape(-1, 1)

            x_i = x[i].reshape(-1, 1)

            dh_next = self.w1.T @ (dl_dh * dh_dz)

            grad_w1 += (dl_dh * dh_dz) @ h_prev.T
            grad_w2 += x_i @ (dl_dh * dh_dz).T
            grad_w3 += dy_dw3 @ dl_dy.T
            grad_b1 += (dl_dh * dh_dz).flatten()
            grad_b2 += dl_dy.flatten()

        self.w1 -= self.lr * np.clip(grad_w1, -5, 5)
        self.w2 -= self.lr * np.clip(grad_w2, -5, 5)
        self.w3 -= self.lr * np.clip(grad_w3, -5, 5)
        self.b1 -= self.lr * np.clip(grad_b1, -5, 5)
        self.b2 -= self.lr * np.clip(grad_b2, -5, 5)

        return loss
