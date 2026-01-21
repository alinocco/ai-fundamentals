import numpy as np

class GRU():
    def _init_weights(self):
        std = np.sqrt(2.0 / self.hidden_size)
        self.w1 = np.random.normal(0, std, size=(self.hidden_size * 3, self.hidden_size))
        self.w2 = np.random.normal(0, std, size=(self.input_size, self.hidden_size * 3))
        self.w3 = np.random.normal(0, std, size=(self.hidden_size, self.output_size))
        self.b1 = np.zeros(self.hidden_size * 3)
        self.b2 = np.zeros(self.output_size)

    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self._init_weights()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x: np.ndarray, hx=None):
        seq_size, _ = x.shape
        if hx is None: hx = np.zeros(self.hidden_size)
        h_t = hx
        
        self.cache = []
        outputs = []

        for i in range(seq_size):
            x_i = x[i]
            
            gate_x = self.w2.T @ x_i
            gate_h = self.w1 @ h_t + self.b1
            
            i_r, i_z, i_n = np.split(gate_x, 3)
            h_r, h_z, h_n = np.split(gate_h, 3)

            r = self.sigmoid(i_r + h_r)
            z = self.sigmoid(i_z + h_z)
            
            n_input = i_n + (r * (self.w1[2*self.hidden_size:] @ h_t + self.b1[2*self.hidden_size:]))
            n = np.tanh(n_input)
            
            h_next = (1 - z) * h_t + z * n
            
            self.cache.append((x_i, h_t, r, z, n, n_input, i_r + h_r, i_z + h_z))
            h_t = h_next
            outputs.append(self.w3.T @ h_t + self.b2)

        return np.array(outputs), h_t

    def backward(self, x: np.ndarray, y: np.ndarray):
        seq_size, _ = x.shape
        _, output_size = y.shape
        outputs, _ = self.forward(x) 
        loss = np.mean((y - outputs) ** 2)

        grad_w1 = np.zeros_like(self.w1)
        grad_w2 = np.zeros_like(self.w2)
        grad_w3 = np.zeros_like(self.w3)
        grad_b1 = np.zeros_like(self.b1)
        grad_b2 = np.zeros_like(self.b2)
        
        dh_next = np.zeros(self.hidden_size)

        for i in reversed(range(seq_size)):
            x_i, h_prev, r, z, n, input_n, input_r, input_z = self.cache[i]
            dy = 2 * (outputs[i] - y[i]) 
            
            h_curr = (1 - z) * h_prev + z * n
            grad_w3 += np.outer(h_curr, dy)
            grad_b2 += dy

            dh = self.w3 @ dy + dh_next
            
            dn = dh * z * (1 - n**2)
            dz = dh * (n - h_prev) * z * (1 - z)
            dr = (self.w1[2*self.hidden_size:].T @ (dn * r)) * h_prev * r * (1 - r)
            
            d_gates = np.concatenate([dr, dz, dn])
            
            d_h_n_comp = dn
            grad_w1[:2*self.hidden_size] += np.outer(d_gates[:2*self.hidden_size], h_prev)
            grad_w1[2*self.hidden_size:] += np.outer(d_h_n_comp, r * h_prev)
            
            grad_b1 += np.concatenate([dr, dz, d_h_n_comp])
            
            grad_w2 += np.outer(x_i, d_gates)
            
            dh_prev_z = dh * (1 - z)
            dh_prev_r = (self.w1[:self.hidden_size].T @ dr) + (self.w1[self.hidden_size:2*self.hidden_size].T @ dz)
            dh_prev_n = self.w1[2*self.hidden_size:].T @ (dn * r)
            
            dh_next = dh_prev_z + dh_prev_r + dh_prev_n

        self.w1 -= self.lr * np.clip(grad_w1, -5, 5)
        self.w2 -= self.lr * np.clip(grad_w2, -5, 5)
        self.w3 -= self.lr * np.clip(grad_w3, -5, 5)
        self.b1 -= self.lr * np.clip(grad_b1, -5, 5)
        self.b2 -= self.lr * np.clip(grad_b2, -5, 5)

        return loss
