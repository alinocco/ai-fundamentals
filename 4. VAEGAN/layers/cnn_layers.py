 
import numpy as np 
import matplotlib.pyplot as plt 
from utils.analysis_tools import cross_entropy,plot_multiclass_roc,calculate_metrics
from utils import epsilon
from numpy.lib.stride_tricks import sliding_window_view
from .basic_layers import Layer
#Свёрточный слой
class ConvolutionLayer(Layer):
    def __init__(self,n_in,n_out,size,stride = 1,c_in = 1,beta1=0.9,beta2=0.95,l2_lambda=0.0005):
        super().__init__()
        std = np.sqrt(2/ n_in  )
        self.core = [np.random.normal(0, std, size=(n_in,size,size)) for i in range(n_out)]
        self.core = np.array(self.core)
        
        self.l2_lambda = l2_lambda

        self.size = size
        self.stride = stride
        self.padding = self.size // 2

        #для adam
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.m_adam = 0
        self.v_adam = 0
        self.t = 0

        self.W_x = []
        self.out_size = None
        self.grad = None
        self.batches = 0
        self.n_in = n_in
        self.n_out = n_out
        self.x_padded = None

        self.M_arr = None

        self.h_out = None
        self.w_out = None
         
 
    def post_init(self, x):
        self.x = x
        N, C, H, W = self.x.shape             # (batch, channels, height, width)
        F, Ck, kH, kW = self.core.shape       # (filters, channels, kernel_height, kernel_width)

        assert C == Ck, f"Channel mismatch: input {C}, kernel {Ck}"

        # Padding по периметру
        self.x_padded = np.pad(
            self.x,
            pad_width=((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode='constant',
            constant_values=0
        )

        # Размер выхода
        self.h_out = (H + 2 * self.padding - kH) // self.stride + 1
        self.w_out = (W + 2 * self.padding - kW) // self.stride + 1
         
        # windows: (N, C, H', W', kH, kW)
        windows = sliding_window_view(self.x_padded, (kH, kW), axis=(2, 3))

        # применяем шаг stride
        windows = windows[:, :, ::self.stride, ::self.stride, :, :]  # (N, C, h_out, w_out, kH, kW)

        # итоговая форма -> (N, h_out*w_out, C*kH*kW)
        self.M_arr = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N, self.h_out * self.w_out, C * kH * kW)


    def forward(self, x):
        self.x = x
        N = x.shape[0]

        W = self.core.reshape(self.n_out, -1).T  # (C*kH*kW, F)

        Y = self.M_arr @ W  # (N, h_out*w_out, F)

        Y = Y.transpose(0, 2, 1).reshape(N, self.n_out, self.h_out, self.w_out)
        return Y

    def backward(self, grad_out):
        N, F, h_out, w_out = grad_out.shape
        C, kH, kW = self.core.shape[1:]
        #градиент по ядру
        grad_flat = grad_out.reshape(N, F, h_out * w_out)
        #как элементы ядра повлияли на ошибку
        dCore_flat = np.einsum('nlk,nfl->fk', self.M_arr, grad_flat)
        dCore = dCore_flat.reshape(F, C, kH, kW)
        self.grad = self.grad + dCore if self.grad is not None else dCore

        #переворачиваем по корреляции
        kernels_rot_flat = np.flip(self.core, axis=(-2, -1)).reshape(F, C*kH*kW)
        dX_flat = np.einsum('nfl,fk->nlk', grad_flat, kernels_rot_flat)
        dX_windows = dX_flat.reshape(N, h_out, w_out, C, kH, kW)

        #всё ниже - это мы восстанавливаем координаты и данные исходного входа
        n_idx = np.repeat(np.arange(N), C*h_out*w_out*kH*kW)
        c_idx = np.tile(np.repeat(np.arange(C), h_out*w_out*kH*kW), N)

        i_idx = np.tile(np.repeat(np.arange(h_out), w_out*kH*kW), N*C)
        j_idx = np.tile(np.tile(np.arange(w_out), h_out*kH*kW), N*C)

        kh_idx = np.tile(np.arange(kH), N*C*h_out*w_out*kW)
        kw_idx = np.tile(np.arange(kW), N*C*h_out*w_out*kH)

        h_idx = i_idx * self.stride + kh_idx
        w_idx = j_idx * self.stride + kw_idx

        #перемешиваем данные и разворачиваем
        dX_vals = dX_windows.transpose(0, 3, 1, 2, 4, 5).ravel()

        dX_padded = np.zeros(self.x_padded.shape, dtype=np.float64)
        np.add.at(dX_padded, (n_idx, c_idx, h_idx, w_idx), dX_vals)

        _, _, H, W = self.x.shape
        dX = dX_padded[:, :, :H, :W]

        self.batches += 1
        return dX



    def update(self,lr):

        self.t += 1

        pre_m_adam = self.beta1 * self.m_adam + (1-self.beta1) * self.grad
        pre_v_adam = self.beta2 * self.v_adam + (1-self.beta2)*self.grad**2

        self.m_adam = pre_m_adam / (1-self.beta1**self.t)
        self.v_adam = pre_v_adam / (1-self.beta2**self.t)

        grad_with_l2 = self.m_adam + self.l2_lambda * self.core

        self.core -= (lr * grad_with_l2 / (np.sqrt(self.v_adam + epsilon))) #/ self.batches

         
        self.grad.fill(0)
        self.batches = 0 
        self.M_arr = []
 

class MaxPoolingLayer(Layer):
    def __init__(self, n_in, n_out, pool_size):
        super().__init__()
        self.size = pool_size
        self.x = None
        self.max_index = None

    def forward(self, x):
        self.x = x
        N,C, H, W = self.x.shape
         
         
         
        out_H, out_W = H // self.size, W // self.size

        result = np.zeros((N,C, out_H, out_W))
        self.max_index = np.zeros((N,C, out_H, out_W), dtype=object)

        # 1. Создаём view всех окон
        windows = sliding_window_view(self.x, (self.size, self.size), axis=(2,3))
        # shape: (N, C, H_p - size + 1, W_p - size + 1, size, size)

        # 2. Применяем stride
        windows = windows[:, :, ::self.size, ::self.size, :, :]  # (N, C, out_H, out_W, size, size)

        # 3. Находим максимальные значения и индексы в окне
        flat_windows = windows.reshape(N, C, out_H, out_W, self.size*self.size)
        max_idx_flat = np.argmax(flat_windows, axis=-1)  # (N, C, out_H, out_W)
        result = np.take_along_axis(flat_windows, max_idx_flat[..., None], axis=-1)[..., 0]  # (N, C, out_H, out_W)

        # 4. Переводим индекс в координаты относительно x_padded
        idx_h = max_idx_flat // self.size
        idx_w = max_idx_flat % self.size

        # 5. Сохраняем реальные координаты
        self.max_index = np.zeros((N, C, out_H, out_W, 2), dtype=int)
        h_start = np.arange(out_H)[:, None] * self.size  # (out_H,1)
        w_start = np.arange(out_W)[None, :] * self.size  # (1,out_W)

        self.max_index[..., 0] = h_start + idx_h  # реальные H координаты
        self.max_index[..., 1] = w_start + idx_w  # реальные W координаты

        return result
    def backward(self, grad_out): 
        N, C, H_out, W_out = grad_out.shape
        dX = np.zeros_like(self.x)  # (N, C, H, W)

        grad_flat = grad_out.ravel()  # (N*C*H_out*W_out, )
    
        max_i = self.max_index[..., 0].ravel()
        max_j = self.max_index[..., 1].ravel()

        # Индексы батча и канала
        batch_idx = np.repeat(np.arange(N), C*H_out*W_out)
        channel_idx = np.tile(np.repeat(np.arange(C), H_out*W_out), N)

        np.add.at(dX, (batch_idx, channel_idx, max_i, max_j), grad_flat)

        return dX


class Sequential():
    def __init__(self,*layers :Layer,loss_function = cross_entropy):
        self.layers = list(layers)
        self.loss_function = loss_function

        self.x = []
        self.losses = []

    def forward(self,x):
        for layer in self.layers: 
            if hasattr(layer,'post_init'):
                layer.post_init(x)
            x = layer.forward(x)
        return x

    def backward(self,grad_out):
        for layer in reversed(self.layers): 
            grad_out = layer.backward(grad_out)
        return grad_out

    def update(self,lr=0.001):
        for layer in reversed(self.layers): 
            if hasattr(layer,'update'):
                layer.update(lr)


    


    def fit(self,train_x,train_y,test_x,test_y,epoches = 200,lr=0.001,batch_size = 64):
        size = train_x.shape[0]
        for i in range(epoches): 
            size = train_x.shape[0] 
            losses_batch = []
            for start in range(0,size,batch_size):
                end = start+batch_size
                x_batch = train_x[start:end]
                y_batch = train_y[start:end]  


              
                y_pred_batch = self.forward(x_batch)
                error = (y_pred_batch - y_batch)/batch_size
                losses_batch.append(self.loss_function(y_pred_batch, y_batch))
                self.backward(error)
                self.update(lr)

            # y_pred_test = self.forward(test_x)
            # self.update(lr)
            loss = np.mean(losses_batch)
            losses_batch.clear()
            self.losses.append(loss)
            # accuracy,precision,recall,f1_score = calculate_metrics(y_pred_test,test_y)
            
            if i % 2 == 0:
                 print(f"Epoch {i}. Loss: {loss:.4f} ")
                # print(f"Epoch {i}. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")

        # Тестирование
        y_pred = self.forward(test_x)
        self.loss = self.loss_function(y_pred,test_y)

        #todo Обобщить
        y_pred_classes = np.argmax(y_pred, axis=1)
        self.accuracy,self.precision,self.recall,self.f1_score = calculate_metrics(y_pred,test_y)
        print('Success!')

    def get_metrics(self):
        return {'loss':self.loss,'accuracy':self.accuracy,'precision':self.precision,'recall':self.recall,'f1_score':self.f1_score}

    # График функции потерь
    def show_losses_plot(self):
        plt.figure(figsize=(6,4))
        plt.plot(self.losses, color='red', lw=2)
        plt.xlabel('Эпоха')
        plt.ylabel('Ошибка')
        plt.title('График функции ошибки по эпохам')
        plt.grid(True)
        plt.show(block=True)



    # ROC - кривая
    def show_roc(self, test_x, test_y):
        y_pred = self.forward(test_x) 
        plot_multiclass_roc(test_y,y_pred)