 
from abc import ABC,abstractmethod
import numpy as cp
import matplotlib.pyplot as plt
from utils.losses import Loss 
from layers.basic_layers import Layer,AddictionalLoss



class SequentialBase(ABC):
    @abstractmethod
    def forward(self, *args,**kwargs):
        pass
    @abstractmethod
    def backward(self, *args,**kwargs):
        pass 
class SequentialFittable(ABC): 
    @abstractmethod
    def fit(self,*args,**kwargs):
        pass

class SequentialAnalyzer(ABC):
    @abstractmethod
    def show_plots(self):
        pass
    @abstractmethod
    def show_metrics(self):
        pass




 
class Sequential(SequentialBase,SequentialAnalyzer):
    def __init__(self, *layers: Layer, loss: Loss):
        self.layers = list(layers)
        self.loss = loss
        self.losses = []
        self.main_loss_batch = []
        self.main_loss_epoch = []
        self.aditionals_losses_batch = []
        self.aditionals_losses_epoch = []
        self.addictional_error = 0
        self.addictional_loss = (0, 0)

    def forward(self, x, cond=None):
        """Прямой проход через все слои с учётом условного вектора и дополнительной ошибки"""
        x = cp.array(x, dtype=cp.float32)
        cond = cp.array(cond, dtype=cp.float32) if cond is not None else None
        if cond is not None:
            x = cp.concatenate([x, cond], axis=1)

        for layer in self.layers:
            x = layer.forward(x, cond=cond) if 'cond' in layer.forward.__code__.co_varnames else layer.forward(x)
            
            if isinstance(layer, AddictionalLoss):
                self.addictional_error += layer.get_addictional_lose()
                self.addictional_loss = layer.get_addictional_lose_grad()

        return x
    def backward(self, grad_out):
        """Обратный проход"""
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def update(self, lr=0.001):
        """Обновить градиенты"""
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(lr)

    def reset(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_state'):
                layer.reset_state()

    def predict(self, x_seq):
        self.addictional_error = 0
        x_seq = cp.array(x_seq, dtype=cp.float32)
        for layer in self.layers:
            x_seq = layer.forward(x_seq)
        return x_seq

    def _shuffle_data(self, x, y, cond=None):
        """Перемешивает данные перед эпохой"""
        n_samples = x.shape[0]
        idx = cp.random.permutation(n_samples)
        x_shuf = x[idx]
        y_shuf = y[idx]
        cond_shuf = cond[idx] if cond is not None else None
        return x_shuf, y_shuf, cond_shuf

    def _train_on_batch(self, x_batch, y_batch, cond_batch, lr):
        """Обучение на одном батче"""
        self.addictional_error = 0
        y_pred = self.forward(x_batch, cond=cond_batch)

        reconstruction_loss = self.loss.loss_func(y_pred, y_batch)
        self.main_loss_batch.append(reconstruction_loss)
        self.aditionals_losses_batch.append(self.addictional_error)

        reconstruction_grad = self.loss.delta_loss_func(y_pred, y_batch)
        self.backward(reconstruction_grad)
        self.update(lr)

        return float(reconstruction_loss)

    def fit(self, train_x_seq, train_y, train_cond=None,
            epoches=200, lr=0.001, batch_size=64):
        
        # Конвертация в CuPy
        train_x_seq = cp.array(train_x_seq, dtype=cp.float32)
        train_y = cp.array(train_y, dtype=cp.float32)
        if train_cond is not None:
            train_cond = cp.array(train_cond, dtype=cp.float32)

        n_samples = train_x_seq.shape[0]

        for epoch in range(epoches):
            x_shuf, y_shuf, cond_shuf = self._shuffle_data(train_x_seq, train_y, train_cond)

            epoch_losses = []
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = x_shuf[start:end]
                y_batch = y_shuf[start:end]
                cond_batch = cond_shuf[start:end] if cond_shuf is not None else None

                batch_loss = self._train_on_batch(x_batch, y_batch, cond_batch, lr)
                epoch_losses.append(batch_loss)

            self.losses.append(float(cp.mean(cp.array(epoch_losses))))
            self.main_loss_epoch.append(float(cp.mean(cp.array(self.main_loss_batch))))
            self.aditionals_losses_epoch.append(float(cp.mean(cp.array(self.aditionals_losses_batch))))

            if epoch % 2 == 0:
                print(f"Epoch {epoch}. Loss: {self.main_loss_epoch[-1]} "
                    f"Dkl-loss {self.aditionals_losses_epoch[-1]}")
                

    def show_plots(self, plot_all=True):
        """
        Построение графиков ошибок.
        Если plot_all=True, строятся все три графика в одной фигуре.
        Иначе строятся три отдельных графика.
        """
        if plot_all:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(self.losses, color='red', lw=2)
            plt.xlabel('Эпоха')
            plt.ylabel('Ошибка')
            plt.title('Общая ошибка')
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(self.main_loss_epoch, color='blue', lw=2)
            plt.xlabel('Эпоха')
            plt.ylabel('Ошибка')
            plt.title('Главная ошибка')
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(self.aditionals_losses_epoch, color='green', lw=2)
            plt.xlabel('Эпоха')
            plt.ylabel('Ошибка')
            plt.title('Доп. ошибка')
            plt.grid(True)

            plt.tight_layout()
            plt.show(block=True)
        else:
            # Можно построить отдельно
            for name, data, color in [
                ("Общая ошибка", self.losses, 'red'),
                ("Главная ошибка", self.main_loss_epoch, 'blue'),
                ("Доп. ошибка", self.aditionals_losses_epoch, 'green')
            ]:
                plt.figure(figsize=(6, 4))
                plt.plot(data, color=color, lw=2)
                plt.xlabel('Эпоха')
                plt.ylabel('Ошибка')
                plt.title(name)
                plt.grid(True)
                plt.show(block=True)

    def show_metrics(self):
        """
        Вывод основных метрик обучения:
        - Последние значения ошибок
        - Среднее по эпохам
        """
        print("=== Metrics ===")
        print(f"Последняя общая ошибка: {self.losses[-1] if self.losses else 0:.6f}")
        print(f"Средняя общая ошибка: {cp.mean(cp.array(self.losses)) if self.losses else 0:.6f}")
        print(f"Последняя главная ошибка: {self.main_loss_epoch[-1] if self.main_loss_epoch else 0:.6f}")
        print(f"Средняя главная ошибка: {cp.mean(cp.array(self.main_loss_epoch)) if self.main_loss_epoch else 0:.6f}")
        print(f"Последняя дополнительная ошибка: {self.aditionals_losses_epoch[-1] if self.aditionals_losses_epoch else 0:.6f}")
        print(f"Средняя дополнительная ошибка: {cp.mean(cp.array(self.aditionals_losses_epoch)) if self.aditionals_losses_epoch else 0:.6f}")
 