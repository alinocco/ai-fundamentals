 
import urllib.request
import os
epsilon = 1e-8
import gzip
import struct
import numpy as cp
  

def load_images(fname):
    with gzip.open(fname, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Неверный формат изображений")
        data = f.read()
        return cp.frombuffer(data, dtype=cp.uint8).reshape(n, rows, cols)

def load_labels(fname):
    with gzip.open(fname, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Неверный формат меток")
        return cp.frombuffer(f.read(), dtype=cp.uint8)

def to_one_hot(y, num_classes=10):
    one_hot = cp.zeros((y.shape[0], num_classes))
    one_hot[cp.arange(y.shape[0]), y] = 1
    return one_hot

def load_base(digit=None, sample_size=None):
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    # === Скачивание при необходимости ===
    for f in files:
        if not os.path.exists(f):
            print("Downloading", f)
            urllib.request.urlretrieve(base + f, f)

    # === Загрузка ===
    X_train = load_images("train-images-idx3-ubyte.gz").astype('float32') / 255.0
    y_train = load_labels("train-labels-idx1-ubyte.gz")
    X_test = load_images("t10k-images-idx3-ubyte.gz").astype('float32') / 255.0
    y_test = load_labels("t10k-labels-idx1-ubyte.gz")

    # === Если задана конкретная цифра — фильтруем ===
    if digit is not None:
        train_mask = (y_train == digit)
        test_mask = (y_test == digit)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

    # === Если задан sample_size — случайно выбираем подвыборку ===
    if sample_size is not None:
        rng = cp.random.default_rng()
        train_indices = rng.choice(len(X_train), min(sample_size, len(X_train)), replace=False)
        test_indices = rng.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]

    # === One-hot для условного признака ===
    y_train_oh = to_one_hot(y_train, 10)
    y_test_oh = to_one_hot(y_test, 10)


# === Конвертация всех данных в CuPy ===
    train_x = cp.array(X_train.reshape(X_train.shape[0], -1).astype(cp.float32) / 255.0)
    test_x = cp.array(X_test.reshape(X_test.shape[0], -1).astype(cp.float32) / 255.0)
    train_y_oh = cp.array(y_train_oh.astype(cp.float32))
    test_y_oh = cp.array(y_test_oh.astype(cp.float32))

    # === Конкатенация через CuPy ===
    train_x_cond = cp.concatenate([train_x, train_y_oh], axis=1)
    test_x_cond = cp.concatenate([test_x, test_y_oh], axis=1)

    return train_x, y_train, train_y_oh, test_x, y_test, test_y_oh,train_x_cond,test_x_cond


 

def calculate_metrics(y_pred,y_true):
    # Преобразуем предсказания в классы
    y_pred_classes = cp.argmax(y_pred, axis=1)
    y_true_classes = cp.argmax(y_true, axis=1)    
    TP = cp.sum((y_pred_classes == 1) & (y_true_classes == 1))
    TN = cp.sum((y_pred_classes == 0) & (y_true_classes == 0))
    FP = cp.sum((y_pred_classes == 1) & (y_true_classes == 0))
    FN = cp.sum((y_pred_classes == 0) & (y_true_classes == 1))
    
    # Метрики
    accuracy = (TP + TN)/(TP + TN + FP + FN + epsilon)
    precision = TP/(TP + FP + epsilon)
    recall = TP/(TP + FN + epsilon)
    f1_score = 2*(precision*recall)/(precision + recall + epsilon)
    fpr = FP/(TN+FP+epsilon)


    return accuracy,precision,recall,f1_score
  
def to_one_hot(y, num_classes=10):
    one_hot = cp.zeros((y.shape[0], num_classes))
    one_hot[cp.arange(y.shape[0]), y] = 1
    return one_hot

def compute_roc(y_true, y_prob):
    thresholds = cp.linspace(0, 1, 100)
    fprs, tprs = [], []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        TP = cp.sum((y_pred==1) & (y_true==1))
        TN = cp.sum((y_pred==0) & (y_true==0))
        FP = cp.sum((y_pred==1) & (y_true==0))
        FN = cp.sum((y_pred==0) & (y_true==1))
        fprs.append(FP/(FP+TN+1e-8))
        tprs.append(TP/(TP+FN+1e-8))
    return cp.array(fprs), cp.array(tprs), thresholds
