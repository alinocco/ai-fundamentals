
import numpy as np
from .basic_layers import Layer,LinearLayer,AddictionalLoss

class MultiLinear(Layer):
    '''
    слой, который принимает много линейных слоев, они обучаются на одном и том же входе,
    отдают вектор одинакового размера и получаются разную ошибку
    '''
    def __init__(self, n_in, n_out,layers_count = 2, l2_lambda=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.layers = []
        for i in range(layers_count):
            self.layers.append(LinearLayer(n_in=n_in,n_out=n_out,l2_lambda=l2_lambda,beta1=beta1,beta2=beta2,epsilon=epsilon))
        self.n_in = n_in
        self.n_out = n_out
        

    def forward(self, x):
        results = []
        for layer in self.layers:
            results.append(layer.forward(x))
        return results
    
    def backward(self, grad_out): 
        if len(grad_out) != len(self.layers):
            raise ValueError(f"grad_out должен быть списком длины {len(self.layers)}")
        errors = []
        for layer, grad in zip(self.layers, grad_out):
            errors.append(layer.backward(grad))
        return sum(errors) 
    
    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

class VAEConstructorLayer(Layer, AddictionalLoss):
    """
    Вызвается после энкодера, чтобы получить параметры латентного распределения,
    посчитать KL-дивергенцию и вернуть скрытое состояние.
    """
    def __init__(self, n_in, n_out, l2_lambda=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, beta_kl=1):
        super().__init__()
        self.multilin = MultiLinear(n_in, n_out, 2, l2_lambda, beta1, beta2, epsilon)
        self.last_epsilon = None
        self.last_dkl = None 
        self.beta_kl = beta_kl   
        self.last_forward = None


    def forward(self, x):
        mu, sigma = self.multilin.forward(x)
         
        self.last_forward = (mu,sigma)
         
        self.last_epsilon = np.random.randn(*mu.shape)
        self.last_dkl = self.calculate_dkl(mu, sigma)

        
        return mu + sigma * self.last_epsilon

    def backward(self, grad_out):
        mu, sigma = self.last_forward
        grad_mu = grad_out + self.beta_kl * mu           # KL по mu
        grad_sigma = grad_out * self.last_epsilon + self.beta_kl * (sigma - 1/sigma)  # KL по sigma
        return self.multilin.backward([grad_mu, grad_sigma])

    def update(self, lr):
        self.multilin.update(lr)

    def calculate_dkl(self, mu, sigma):
        """
        KL(N(mu,sigma^2) || N(0,1))
        """
        sigma2 = sigma**2
        dkl = -0.5 * np.sum(1 + np.log(sigma2) - mu**2 - sigma2, axis=1)
        return np.mean(dkl)
    
    def get_addictional_lose_grad(self):
        mu, sigma = self.last_forward
        grad_mu = self.beta_kl * mu
        grad_sigma = self.beta_kl * (sigma - 1/sigma)
        return grad_mu, grad_sigma

    def get_addictional_lose(self):
        if self.last_dkl is None:
            raise ValueError('Not valid state for KL')
         
        # β-VAE
        return self.beta_kl * self.last_dkl  
    
class CVAEConstructorLayer(VAEConstructorLayer):
    def forward(self, x, cond=None,inference = False):
        """
        Условный аналог VAEConstructorLayer
        x: выход предыдущего слоя энкодера
        cond: one-hot метка класса
        """
        if inference:
            if cond is not None:
                return np.concatenate([x, cond], axis=1)
            return x
        self.last_cond = cond
        if cond is not None:
            x = np.concatenate([x, cond], axis=1)   

        # Получаем параметры латентного распределения
        mu, sigma = self.multilin.forward(x)
        mu = np.clip(mu, -3, 3)
        sigma = np.clip(sigma, 1e-2, 3)  

        self.last_forward = (mu, sigma)
        self.last_epsilon = np.random.randn(*mu.shape)
        self.last_dkl = self.calculate_dkl(mu, sigma)

        z = mu + sigma * self.last_epsilon
 
        return z

    def backward(self, grad_out):
        """
        grad_out: градиент от декодера (batch, latent+cond)
        """
        if self.last_cond is not None:
            cond_dim = self.last_cond.shape[1]
            grad_z = grad_out[:, :-cond_dim]  # берем только латентную часть
        else:
            grad_z = grad_out
            

        mu, sigma = self.last_forward

        # Градиенты β-VAE
        grad_mu = grad_z + self.beta_kl * mu           # KL по μ
        grad_sigma = grad_z * self.last_epsilon + self.beta_kl * (sigma - 1/sigma)  # KL по σ

        grad_input = self.multilin.backward([grad_mu, grad_sigma])

        if self.last_cond is not None:
            grad_input = grad_input[:, :-cond_dim]

        return grad_input

    def get_addictional_lose(self):
        """Возвращает β-KL для текущего батча"""
        if self.last_dkl is None:
            raise ValueError("Not valid state for KL")
        return self.beta_kl * self.last_dkl
