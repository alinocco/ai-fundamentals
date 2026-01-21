import numpy as np

class LSTM():
    def _init_weights(self):
        std = np.sqrt(2.0 / self.hidden_size)
        self.w1 = np.random.normal(0, std, size=(self.hidden_size * 4, self.hidden_size))
        self.w2 = np.random.normal(0, std, size=(self.input_size, self.hidden_size * 4))
        self.w3 = np.random.normal(0, std, size=(self.hidden_size, self.output_size))
        self.b1 = np.zeros(self.hidden_size * 4)
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
        if hx is None:
            h_t = np.zeros(self.hidden_size)
            c_t = np.zeros(self.hidden_size)
        else:
            h_t, c_t = hx
        
        self.cache = []
        outputs = []
        
        for i in range(seq_size):
            x_i = x[i]
            gates = self.w1 @ h_t + self.w2.T @ x_i + self.b1
            
            i_g, f_g, o_g, c_tilde_g = np.split(gates, 4)
            
            i_t = self.sigmoid(i_g)
            f_t = self.sigmoid(f_g)
            o_t = self.sigmoid(o_g)
            c_tilde = np.tanh(c_tilde_g)
            
            c_next = f_t * c_t + i_t * c_tilde
            h_next = o_t * np.tanh(c_next)
            
            self.cache.append((x_i, h_t, c_t, i_t, f_t, o_t, c_tilde, c_next, h_next))
            
            h_t = h_next
            c_t = c_next
            outputs.append(self.w3.T @ h_t + self.b2)
            
        return np.array(outputs), (h_t, c_t)

    def backward(self, x: np.ndarray, y: np.ndarray):
        seq_size, _ = x.shape
        outputs, _ = self.forward(x)
        loss = np.mean((y - outputs) ** 2)
        
        grad_w1 = np.zeros_like(self.w1)
        grad_w2 = np.zeros_like(self.w2)
        grad_w3 = np.zeros_like(self.w3)
        grad_b1 = np.zeros_like(self.b1)
        grad_b2 = np.zeros_like(self.b2)
        
        dh_next = np.zeros(self.hidden_size)
        dc_next = np.zeros(self.hidden_size)
        
        for i in reversed(range(seq_size)):
            x_i, h_prev, c_prev, i_t, f_t, o_t, c_tilde, c_curr, h_curr = self.cache[i]
            
            dy = 2 * (outputs[i] - y[i])
            grad_w3 += np.outer(h_curr, dy)
            grad_b2 += dy
            
            dh = self.w3 @ dy + dh_next
            
            do = dh * np.tanh(c_curr)
            do_raw = do * o_t * (1 - o_t)
            
            dc = dh * o_t * (1 - np.tanh(c_curr)**2) + dc_next
            
            dc_tilde = dc * i_t
            dc_tilde_raw = dc_tilde * (1 - c_tilde**2)
            
            di = dc * c_tilde
            di_raw = di * i_t * (1 - i_t)
            
            df = dc * c_prev
            df_raw = df * f_t * (1 - f_t)
            
            d_gates = np.concatenate([di_raw, df_raw, do_raw, dc_tilde_raw])
            
            grad_w1 += np.outer(d_gates, h_prev)
            grad_w2 += np.outer(x_i, d_gates)
            grad_b1 += d_gates
            
            dh_next = self.w1.T @ d_gates
            dc_next = dc * f_t
            
        self.w1 -= self.lr * np.clip(grad_w1, -5, 5)
        self.w2 -= self.lr * np.clip(grad_w2, -5, 5)
        self.w3 -= self.lr * np.clip(grad_w3, -5, 5)
        self.b1 -= self.lr * np.clip(grad_b1, -5, 5)
        self.b2 -= self.lr * np.clip(grad_b2, -5, 5)
        
        return loss
