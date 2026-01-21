 
import numpy as np 
from .basic_layers import Layer
 

import numpy as np
from .basic_layers import Layer

class RNNLayer(Layer):
    def __init__(self, n_in: int, n_out: int, 
                 activation_func=np.tanh, 
                 activation_func_delta=lambda x: 1 - x**2,
                 l2_lambda=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation_func
        self.activation_grad = activation_func_delta
        
        # Инициализация Хе (для tanh) или Глорота
        std = np.sqrt(2 / (n_in + n_out))
        self.Wx = np.random.normal(0, std, size=(n_in, n_out))   # вход -> скрытое
        self.Wh = np.random.normal(0, std, size=(n_out, n_out))  # скрытое -> скрытое
        self.b = np.zeros((1, n_out))
        
        # Градиенты
        self.grad_Wx = np.zeros_like(self.Wx)
        self.grad_Wh = np.zeros_like(self.Wh)
        self.grad_b = np.zeros_like(self.b)
        
        # Adam
        self.m_Wx = np.zeros_like(self.Wx); self.v_Wx = np.zeros_like(self.Wx)
        self.m_Wh = np.zeros_like(self.Wh); self.v_Wh = np.zeros_like(self.Wh)
        self.m_b  = np.zeros_like(self.b);  self.v_b  = np.zeros_like(self.b)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.t = 0
        self.l2_lambda = l2_lambda
        
        # Кэш для backward
        self.h_states = []  # [h0, h1, ..., hT]
        self.x_inputs = []  # [x0, x1, ..., xT]

    def reset_state(self):
        self.h_states = []
        self.x_inputs = []

    def post_init(self, x):
        # Вызывается в Sequential.forward при первом проходе
        if not hasattr(self, 'h_prev') or self.h_prev is None:
            batch_size = x.shape[0]
            self.h_prev = np.zeros((batch_size, self.n_out))

    def forward(self, x_seq):
        # x_seq: (batch_size, time_steps, input_dim)
        batch_size, T, _ = x_seq.shape
        h = np.zeros((batch_size, self.n_out))
        self.h_states = []
        self.x_inputs = []

        for t in range(T):
            x_t = x_seq[:, t, :]  # (B, n_in)
            h = self.activation(np.dot(x_t, self.Wx) + np.dot(h, self.Wh) + self.b)
            self.h_states.append(h.copy())
            self.x_inputs.append(x_t.copy())

        return h

    def backward(self, grad_out):
        # grad_out: (batch_size, n_out) — градиент по h_t
        if len(self.h_states) == 0:
            raise RuntimeError("Backward called before forward or after reset")

        # Итерируем по времени в обратном порядке
        grad_h_next = grad_out
        grad_Wx = np.zeros_like(self.Wx)
        grad_Wh = np.zeros_like(self.Wh)
        grad_b = np.zeros_like(self.b)
        grad_x = None

        T = len(self.h_states)
        for t in reversed(range(T)):
            x_t = self.x_inputs[t]
            h_t = self.h_states[t]
            h_prev = np.zeros_like(h_t) if t == 0 else self.h_states[t-1]

            # Градиент по активации
            delta = grad_h_next * self.activation_grad(h_t)

            # Накопление градиентов
            grad_Wx += x_t.T @ delta
            grad_Wh += h_prev.T @ delta
            grad_b += np.sum(delta, axis=0, keepdims=True)

            # Градиент по входу x_t
            grad_x_t = delta @ self.Wx.T

            # Градиент по предыдущему скрытому состоянию
            if t > 0:
                grad_h_next = delta @ self.Wh.T
            else:
                grad_h_next = None

            grad_x = grad_x_t

        # Сохраняем градиенты для update
        self.grad_Wx = grad_Wx
        self.grad_Wh = grad_Wh
        self.grad_b = grad_b

        return grad_x  # градиент по первому x (для нижних слоёв)

    def update(self, lr):
        self.t += 1
        
        # Adam для Wx
        self.m_Wx = self.beta1 * self.m_Wx + (1 - self.beta1) * self.grad_Wx
        self.v_Wx = self.beta2 * self.v_Wx + (1 - self.beta2) * (self.grad_Wx ** 2)
        m_hat = self.m_Wx / (1 - self.beta1 ** self.t)
        v_hat = self.v_Wx / (1 - self.beta2 ** self.t)
        self.Wx -= lr * (m_hat + self.l2_lambda * self.Wx) / (np.sqrt(v_hat) + self.epsilon)
        
        # Adam для Wh
        self.m_Wh = self.beta1 * self.m_Wh + (1 - self.beta1) * self.grad_Wh
        self.v_Wh = self.beta2 * self.v_Wh + (1 - self.beta2) * (self.grad_Wh ** 2)
        m_hat = self.m_Wh / (1 - self.beta1 ** self.t)
        v_hat = self.v_Wh / (1 - self.beta2 ** self.t)
        self.Wh -= lr * (m_hat + self.l2_lambda * self.Wh) / (np.sqrt(v_hat) + self.epsilon)
        
        # Adam для b
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * self.grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (self.grad_b ** 2)
        m_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_hat = self.v_b / (1 - self.beta2 ** self.t)
        self.b -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class LSTMLayer(Layer):
    def __init__(self, n_in: int, n_out: int,
                 l2_lambda=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        # Инициализация весов для всех гейтов
        std = np.sqrt(2 / (n_in + n_out))
        self.Wf = np.random.normal(0, std, (n_in, n_out))
        self.Wi = np.random.normal(0, std, (n_in, n_out))
        self.Wc = np.random.normal(0, std, (n_in, n_out))
        self.Wo = np.random.normal(0, std, (n_in, n_out))

        self.Uf = np.random.normal(0, std, (n_out, n_out))
        self.Ui = np.random.normal(0, std, (n_out, n_out))
        self.Uc = np.random.normal(0, std, (n_out, n_out))
        self.Uo = np.random.normal(0, std, (n_out, n_out))

        self.bf = np.zeros((1, n_out))
        self.bi = np.zeros((1, n_out))
        self.bc = np.zeros((1, n_out))
        self.bo = np.zeros((1, n_out))

        # Градиенты
        self.grad = {k: np.zeros_like(v) for k,v in self.__dict__.items() if k.startswith(('W','U','b'))}

        # Adam
        self.m = {k: np.zeros_like(v) for k,v in self.grad.items()}
        self.v = {k: np.zeros_like(v) for k,v in self.grad.items()}

        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.t = 0
        self.l2_lambda = l2_lambda

        # Кэш для backward
        self.h_states = []
        self.c_states = []
        self.x_inputs = []
        self.f_states = []
        self.i_states = []
        self.o_states = []
        self.c_tilde_states = []

    def reset_state(self):
        self.h_states = []
        self.c_states = []
        self.x_inputs = []
        self.f_states = []
        self.i_states = []
        self.o_states = []
        self.c_tilde_states = []

    def post_init(self, x):
        if not hasattr(self, 'h_prev') or self.h_prev is None:
            batch_size = x.shape[0]
            self.h_prev = np.zeros((batch_size, self.n_out))
            self.c_prev = np.zeros((batch_size, self.n_out))

    @staticmethod
    def _sigmoid(x):
        # Ограничиваем диапазон для безопасности
        x = np.clip(x, -500, 500)
        # Устойчивое вычисление сигмоиды
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def forward(self, x_seq):
        batch_size, T, _ = x_seq.shape
        h = np.zeros((batch_size, self.n_out))
        c = np.zeros((batch_size, self.n_out))

        self.reset_state()

        for t in range(T):
            x_t = x_seq[:, t, :]

            f_t = self._sigmoid(np.dot(x_t, self.Wf) + np.dot(h, self.Uf) + self.bf)
            i_t = self._sigmoid(np.dot(x_t, self.Wi) + np.dot(h, self.Ui) + self.bi)
            o_t = self._sigmoid(np.dot(x_t, self.Wo) + np.dot(h, self.Uo) + self.bo)
            c_tilde = np.tanh(np.dot(x_t, self.Wc) + np.dot(h, self.Uc) + self.bc)

            c = f_t * c + i_t * c_tilde
            h = o_t * np.tanh(c)

            self.h_states.append(h.copy())
            self.c_states.append(c.copy())
            self.x_inputs.append(x_t.copy())
            self.f_states.append(f_t.copy())
            self.i_states.append(i_t.copy())
            self.o_states.append(o_t.copy())
            self.c_tilde_states.append(c_tilde.copy())

        self.h_prev = h
        self.c_prev = c
        return h

    def backward(self, grad_out):
        T = len(self.h_states)
        batch_size = grad_out.shape[0]

        grad_h_next = grad_out
        grad_c_next = np.zeros_like(grad_h_next)

        grad_x = np.zeros((batch_size, self.n_in))
        grad = {k: np.zeros_like(v) for k,v in self.grad.items()}

        for t in reversed(range(T)):
            h = self.h_states[t]
            c = self.c_states[t]
            x = self.x_inputs[t]
            f = self.f_states[t]
            i = self.i_states[t]
            o = self.o_states[t]
            c_tilde = self.c_tilde_states[t]
            h_prev = np.zeros_like(h) if t == 0 else self.h_states[t-1]
            c_prev = np.zeros_like(c) if t == 0 else self.c_states[t-1]

            # Градиенты output
            dh = grad_h_next
            do = dh * np.tanh(c)
            do_raw = do * o * (1 - o)

            dc = dh * o * (1 - np.tanh(c)**2) + grad_c_next
            dc_tilde = dc * i
            dc_tilde_raw = dc_tilde * (1 - c_tilde**2)

            di = dc * c_tilde
            di_raw = di * i * (1 - i)

            df = dc * c_prev
            df_raw = df * f * (1 - f)

            # Градиенты по входу x_t
            grad_x += df_raw @ self.Wf.T + di_raw @ self.Wi.T + dc_tilde_raw @ self.Wc.T + do_raw @ self.Wo.T

            # Градиенты по весам и смещениям
            grad['Wf'] += x.T @ df_raw
            grad['Wi'] += x.T @ di_raw
            grad['Wc'] += x.T @ dc_tilde_raw
            grad['Wo'] += x.T @ do_raw

            grad['Uf'] += h_prev.T @ df_raw
            grad['Ui'] += h_prev.T @ di_raw
            grad['Uc'] += h_prev.T @ dc_tilde_raw
            grad['Uo'] += h_prev.T @ do_raw

            grad['bf'] += np.sum(df_raw, axis=0, keepdims=True)
            grad['bi'] += np.sum(di_raw, axis=0, keepdims=True)
            grad['bc'] += np.sum(dc_tilde_raw, axis=0, keepdims=True)
            grad['bo'] += np.sum(do_raw, axis=0, keepdims=True)

            # Подготовка к следующему шагу по времени
            grad_h_next = df_raw @ self.Uf.T + di_raw @ self.Ui.T + dc_tilde_raw @ self.Uc.T + do_raw @ self.Uo.T
            grad_c_next = dc * f

        self.grad = grad
        return grad_x

    def update(self, lr):
        self.t += 1
        for k in self.grad.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * self.grad[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (self.grad[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.__dict__[k] -= lr * (m_hat + self.l2_lambda * self.__dict__[k]) / (np.sqrt(v_hat) + self.epsilon)
            self.grad[k].fill(0)


class GRULayer(Layer):
    def __init__(self, n_in: int, n_out: int,
                 l2_lambda=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        std = np.sqrt(2 / (n_in + n_out))
        # Веса для update, reset и candidate
        self.Wz = np.random.normal(0, std, (n_in, n_out))
        self.Wr = np.random.normal(0, std, (n_in, n_out))
        self.Wh = np.random.normal(0, std, (n_in, n_out))

        self.Uz = np.random.normal(0, std, (n_out, n_out))
        self.Ur = np.random.normal(0, std, (n_out, n_out))
        self.Uh = np.random.normal(0, std, (n_out, n_out))

        self.bz = np.zeros((1, n_out))
        self.br = np.zeros((1, n_out))
        self.bh = np.zeros((1, n_out))

        self.grad = {k: np.zeros_like(v) for k,v in self.__dict__.items() if k.startswith(('W','U','b'))}
        self.m = {k: np.zeros_like(v) for k,v in self.grad.items()}
        self.v = {k: np.zeros_like(v) for k,v in self.grad.items()}

        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.t = 0
        self.l2_lambda = l2_lambda

        # Кэш для backward
        self.h_states = []
        self.x_inputs = []
        self.z_states = []
        self.r_states = []
        self.h_tilde_states = []

    def reset_state(self):
        self.h_states = []
        self.x_inputs = []
        self.z_states = []
        self.r_states = []
        self.h_tilde_states = []

    @staticmethod
    def _sigmoid(x):
        # Ограничиваем диапазон для безопасности
        x = np.clip(x, -500, 500)
        # Устойчивое вычисление сигмоиды
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    def forward(self, x_seq):
        batch_size, T, _ = x_seq.shape
        h = np.zeros((batch_size, self.n_out))

        self.reset_state()

        for t in range(T):
            x_t = x_seq[:, t, :]

            z_t = self._sigmoid(np.dot(x_t, self.Wz) + np.dot(h, self.Uz) + self.bz)
            r_t = self._sigmoid(np.dot(x_t, self.Wr) + np.dot(h, self.Ur) + self.br)
            h_tilde = np.tanh(np.dot(x_t, self.Wh) + np.dot(r_t * h, self.Uh) + self.bh)
            h = (1 - z_t) * h + z_t * h_tilde

            self.h_states.append(h.copy())
            self.x_inputs.append(x_t.copy())
            self.z_states.append(z_t.copy())
            self.r_states.append(r_t.copy())
            self.h_tilde_states.append(h_tilde.copy())

        return h

    def backward(self, grad_out):
        T = len(self.h_states)
        batch_size = grad_out.shape[0]

        grad_h_next = grad_out
        grad_x = np.zeros((batch_size, self.n_in))
        grad = {k: np.zeros_like(v) for k,v in self.grad.items()}

        for t in reversed(range(T)):
            h = self.h_states[t]
            h_prev = np.zeros_like(h) if t == 0 else self.h_states[t-1]
            x = self.x_inputs[t]
            z = self.z_states[t]
            r = self.r_states[t]
            h_tilde = self.h_tilde_states[t]

            dh = grad_h_next

            # Градиенты по z, h_tilde
            dz = dh * (h_tilde - h_prev) * z * (1 - z)
            dh_tilde = dh * z * (1 - h_tilde**2)
            dr = (dh_tilde @ self.Uh.T) * h_prev * r * (1 - r)

            # Градиенты по входу x_t
            grad_x += dz @ self.Wz.T + dr @ self.Wr.T + dh_tilde @ self.Wh.T

            # Градиенты по весам
            grad['Wz'] += x.T @ dz
            grad['Wr'] += x.T @ dr
            grad['Wh'] += x.T @ dh_tilde

            grad['Uz'] += h_prev.T @ dz
            grad['Ur'] += h_prev.T @ dr
            grad['Uh'] += (r * h_prev).T @ dh_tilde

            grad['bz'] += np.sum(dz, axis=0, keepdims=True)
            grad['br'] += np.sum(dr, axis=0, keepdims=True)
            grad['bh'] += np.sum(dh_tilde, axis=0, keepdims=True)

            grad_h_next = dh * (1 - z) + dh_tilde @ self.Uh.T * r + dz @ self.Uz.T + dr @ self.Ur.T

        self.grad = grad
        return grad_x

    def update(self, lr):
        self.t += 1
        for k in self.grad.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * self.grad[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (self.grad[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.__dict__[k] -= lr * (m_hat + self.l2_lambda * self.__dict__[k]) / (np.sqrt(v_hat) + self.epsilon)
            self.grad[k].fill(0)

