import numpy as cp
import matplotlib.pyplot as plt
from abc import ABC,abstractmethod
epsilon = 1e-8


def logloss(y_pred,y_true):
     return -cp.mean(y_true*cp.log(y_pred + epsilon) + (1-y_true)*cp.log(1-y_pred + epsilon))

def cross_entropy(y_pred,y_true):
    return -cp.mean(cp.sum(y_true * cp.log(y_pred + epsilon), axis=1))

def cross_entropy_det(y_pred,y_true):
    return y_pred - y_true

def mse(y_pred, y_true): 
    return cp.mean((y_pred - y_true) ** 2)
def mse_grad(y_pred, y_true): 
    return 2 * (y_pred - y_true) / y_pred.shape[0]

class Loss(ABC):
    @abstractmethod
    def loss_func(self,y_pred,y_true):
        pass
    @abstractmethod
    def delta_loss_func(self,y_pred,y_true):
        pass

class CrossEntropy(Loss):
    def loss_func(self,y_pred,y_true):
        return -cp.mean(cp.sum(y_true * cp.log(y_pred + epsilon), axis=1))
    def delta_loss_func(self,y_pred,y_true):
        return y_pred - y_true  

class MSE(Loss):
    def loss_func(self,y_pred,y_true):
        return cp.mean((y_pred - y_true) ** 2)
    def delta_loss_func(self,y_pred,y_true):
        return 2 * (y_pred - y_true) / y_pred.shape[0]


class BCE(Loss):
    def loss_func(self, y_pred, y_true):
        return -cp.mean(y_true*cp.log(y_pred+epsilon) + (1-y_true)*cp.log(1-y_pred+epsilon))

    def delta_loss_func(self, y_pred, y_true):
        return (y_pred - y_true) / (y_pred*(1-y_pred) + epsilon)



def rmse(y_pred, y_true):
    return cp.sqrt(cp.mean((y_pred - y_true) ** 2))

def r2(y_pred, y_true): 
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    ss_res = cp.sum((y_true - y_pred) ** 2)                     
    ss_tot = cp.sum((y_true - cp.mean(y_true)) ** 2)            
    return 1 - (ss_res / ss_tot)

