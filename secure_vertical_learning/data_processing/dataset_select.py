import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix

def select_dataset(X, Y, m, m_selected):
    """ 
    For datasets with excessively large training sample sizes, randomly select a portion of samples as the training set.
    
    Args:
        X: feature matrix
        Y: label matrix
        m: Original sample size
        m_selected: Expected sample size after selection

    Random seed = 1
    """
    assert X.shape[0] == m
    assert len(Y) == m
    np.random.seed(1)
    index = np.random.choice(m, m_selected, replace=False)
    index.sort()
    assert len(index) == m_selected
    X_selected = X[index, :]
    Y_selected = Y[index]
    return X_selected, Y_selected

def load_dataset(name):
    pass

def dataset_selector(dataset_name, train_data, test_data, Y_train):
    """ 
    According to different datasets, select the number of samples to be extracted and return the results.
    Do not convert to dense format, keep sparse. Convert later if needed.

    Args:
        dataset_name: 
            Name of the dataset
        train_data: 
            Training data in sparse matrix format, including features and labels
        Y_train: 
            Labels extracted from train_data
        keep_sparse: 
            For high-dimensional sparse large datasets, keep sparse format to avoid memory overflow

    Returns: Usable sparse format training features and labels
    """
    X_train = train_data[0]
    X_test = test_data[0]

    if dataset_name == "cifar10":
        # X_train = train_data[0].todense().A
        X_train = train_data[0]
        X_train, Y_train = select_dataset(X_train, Y_train, 50000, 10000)
        
    elif dataset_name == "SVHN":
        # X_train = train_data[0].todense().A
        X_train = train_data[0]
        X_train, Y_train = select_dataset(X_train, Y_train, 34750, 10000)
    
    elif dataset_name == "ledgar":
        X_train = train_data[0]
        X_train, Y_train = select_dataset(X_train, Y_train, 70000, 10000)

    elif dataset_name == "webspam10k_50k":
        X_train = csr_matrix(X_train, shape=(X_train.shape[0], 50000))
        X_test = csr_matrix(X_test, shape=(X_test.shape[0], 50000))

    elif dataset_name == "webspam10k_100k":
        X_train = csr_matrix(X_train, shape=(X_train.shape[0], 100000))
        X_test = csr_matrix(X_test, shape=(X_test.shape[0], 100000))

    elif dataset_name == "webspam10k_500k":
        X_train = csr_matrix(X_train, shape=(X_train.shape[0], 500000))
        X_test = csr_matrix(X_test, shape=(X_test.shape[0], 500000))

    return X_train, X_test, Y_train



if __name__ == '__main__':
    dataset_name = "cifar10"
    X, Y = load_dataset(dataset_name)

    if dataset_name == "cifar10":
        """ Train: (50000, 3072) Test: (50000,) """
        select_dataset(X, Y, 50000, 10000)