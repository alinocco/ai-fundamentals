import numpy as cp
import matplotlib.pyplot as plt
from abc import ABC
from layers.vae_layers import CVAEConstructorLayer
epsilon = 1e-8

 

def tSNE(x, perplexity=30, learning_rate=200, n_iter=1000, early_exaggeration=12.0, momentum_init=0.5, momentum_final=0.9):
    n = x.shape[0]

    #считаем расстояния 
    sum_X = cp.sum(cp.square(x), axis=1)
    distances = sum_X[:, None] + sum_X[None, :] - 2 * x.dot(x.T)

    #инициализируем матрицу
    P = cp.zeros((n, n))
    log_perp = cp.log(perplexity)

    #подбор P c нужной beta
    for i in range(n):
        beta_min, beta_max = -cp.inf, cp.inf
        beta = 1.0
        for _ in range(50):
            dists = distances[i, :].copy()
            dists[i] = cp.inf
            p_i = cp.exp(-dists * beta)
            p_i[i] = 0
            p_i /= cp.sum(p_i) + 1e-12
            H = -cp.sum(p_i * cp.log(p_i + 1e-12))
            H_diff = H - log_perp
            if cp.abs(H_diff) < 1e-5:
                break
            if H_diff > 0:
                beta_min = beta
                beta = beta * 2 if beta_max == cp.inf else (beta + beta_max)/2
            else:
                beta_max = beta
                beta = beta / 2 if beta_min == -cp.inf else (beta + beta_min)/2
        P[i, :] = p_i

 
    P = (P + P.T)
    P /= cp.sum(P)
    P *= early_exaggeration

    y = cp.random.randn(n, 2) * 1e-2
    y_update_prev = cp.zeros_like(y)

    for epoch in range(n_iter):
        #два значения момента чтобы градиенты не затухали
        alpha = momentum_init if epoch < 250 else momentum_final

        #   4. Q  
        sum_Y = cp.sum(cp.square(y), axis=1)
        distances_y = sum_Y[:, None] + sum_Y[None, :] - 2 * y.dot(y.T)
        inv_dist = 1.0 / (1.0 + distances_y)
        cp.fill_diagonal(inv_dist, 0)
        Q = inv_dist / cp.sum(inv_dist)
        Q = cp.maximum(Q, 1e-12)

        #   5. Градиент  
        grad = cp.zeros_like(y)
        for i in range(n):
            diff = y[i] - y
            grad[i] = 4 * cp.sum((P[i, :] - Q[i, :])[:, None] * diff * inv_dist[i, :, None], axis=0)

        #  6. Momentum update 
        y_update = alpha * y_update_prev - learning_rate * grad
        y += y_update
        y_update_prev = y_update

        #   7. Снимаем раннее усиление после 100 эпох  
        if epoch == 100:
            P /= early_exaggeration

        if epoch % 100 == 0:
            kl_div = cp.sum(P * cp.log((P + 1e-12) / (Q + 1e-12)))
            print(f"Epoch {epoch}, KL-divergence: {kl_div:.6f}")

    return y

def tSNE_plot(y, labels=None):
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(y[:,0], y[:,1], c=labels, cmap='tab10', s=8, alpha=0.7)
    if labels is not None:
        plt.colorbar(scatter, label='Class')
    plt.title("t-SNE projection")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()




def generate_digit(vae, digit, latent_dim=64):
    import numpy as cp

    z = cp.random.randn(1, latent_dim).astype(cp.float32)

    cond = cp.zeros((1, 10), dtype=cp.float32)
    cond[0, digit] = 1000

    z_cond = cp.concatenate([z, cond], axis=1)

    x = z_cond
    for layer in vae.decoder.layers:
        if hasattr(layer, 'forward') and 'cond' in layer.forward.__code__.co_varnames:
            x = layer.forward(x, cond=cond)
        else:
            x = layer.forward(x)

    return cp.array(x.reshape(28, 28))


 

 
def calculate_metrics(y_pred, y_true):
    y_true_labels = cp.argmax(y_true, axis=1)
    y_pred_labels = cp.argmax(y_pred, axis=1)
    
    k = y_true.shape[1]
    miss_matrix = cp.zeros((k, k), dtype=int)
    for t, p in zip(y_true_labels, y_pred_labels):
        miss_matrix[t, p] += 1

    accuracy = cp.trace(miss_matrix) / cp.sum(miss_matrix)
    
    precision_per_class = cp.zeros(k)
    recall_per_class = cp.zeros(k)
    f1_per_class = cp.zeros(k)

    for i in range(k):
        TP = miss_matrix[i, i]
        FP = cp.sum(miss_matrix[:, i]) - TP
        FN = cp.sum(miss_matrix[i, :]) - TP

        precision_per_class[i] = TP / (TP + FP + 1e-8)
        recall_per_class[i] = TP / (TP + FN + 1e-8)
        f1_per_class[i] = 2 * precision_per_class[i] * recall_per_class[i] / (precision_per_class[i] + recall_per_class[i] + 1e-8)
 
    precision = cp.mean(precision_per_class)
    recall = cp.mean(recall_per_class)
    f1 = cp.mean(f1_per_class)

    print('MissMatrix:')
    print(miss_matrix)
 
    return accuracy, precision, recall, f1

  