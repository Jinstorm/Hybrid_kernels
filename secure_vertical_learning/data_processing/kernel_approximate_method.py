import numpy as np
from sklearn.kernel_approximation import PolynomialCountSketch, Nystroem, RBFSampler

def cal_X_var(X):
    """
    Calculate the variance of the dataset-sparse matrix
    """
    X_mean = X.mean()
    if isinstance(X, np.ndarray):
        X_square = np.multiply(X, X)
    else:
        X_square = X.multiply(X)
    
    X_var = X_square.mean() - X_mean ** 2

    return X_var

def dataSketch_generator_RBFandPolyKernel(X_train, X_test, Y_train, Y_test, kernel_method, sampling_k, partition):
    """
    Use two kernel approximation methods, RBF and Poly — RFF and TensorSketch, to process the dataset
    Return the processed dataset and parameters of the approximation method: gamma_scale
    """
    k = int(sampling_k)
    k1 = np.floor(k * partition).astype(int)
    k2 = k - k1

    n_train = X_train.shape[1]
    n1 = np.floor(n_train * partition).astype(int)
    X_train1, X_train2 = X_train[:,0:n1], X_train[:,n1:]

    n_test = X_test.shape[1]
    n1 = np.floor(n_test * partition).astype(int)
    X_test1, X_test2 = X_test[:,0:n1], X_test[:,n1:]

    gamma_scale = 1. / X_train.shape[1] / cal_X_var(X_train)

    if kernel_method == "rff":
        rff1 = RBFSampler(gamma=gamma_scale, n_components=k1, random_state=1)
        rff2 = RBFSampler(gamma=gamma_scale, n_components=k2, random_state=1)
        X1_train_sketch = rff1.fit_transform(X_train1)
        X1_test_sketch = rff1.fit_transform(X_test1)
        X2_train_sketch = rff2.fit_transform(X_train2)
        X2_test_sketch = rff2.fit_transform(X_test2)

    elif kernel_method == "poly":
        ts1 = PolynomialCountSketch(degree=2, gamma=gamma_scale, coef0=1, n_components=k1, random_state=1)
        ts2 = PolynomialCountSketch(degree=2, gamma=gamma_scale, coef0=1, n_components=k2, random_state=1)
        X1_train_sketch = ts1.fit_transform(X_train1)
        X1_test_sketch = ts1.fit_transform(X_test1)
        X2_train_sketch = ts2.fit_transform(X_train2)
        X2_test_sketch = ts2.fit_transform(X_test2)
    
    print("data generation ({}) ok.".format(kernel_method))
    return X1_train_sketch, X2_train_sketch, Y_train, X1_test_sketch, X2_test_sketch, Y_test, gamma_scale

if __name__ == "__main__":
    pass