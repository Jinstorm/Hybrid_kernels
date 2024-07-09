
import numpy as np
from sklearn.datasets import load_svmlight_file 
import scipy.sparse as sp 
import os
import datetime
import time
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from liblinear.liblinearutil import *


def util_feature(X, portion=0.5):
    if isinstance(X, sp.csc_matrix) == False:
        X = sp.csc_matrix(X)
    p1 = np.floor(X.shape[1] * portion).astype('int')  
    X1 = X[:, 0:p1]  
    X2 = X[:, p1:]
    return X1, X2


def util_feature2(X, n1):
    if isinstance(X, sp.csc_matrix) == False:
        X = sp.csc_matrix(X)
    X1 = X[:, 0:n1] 
    X2 = X[:, n1:]
    return X1, X2

def cal_X_b(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])  



def cal_X_b_v2(X):
    return sp.hstack(
        [sp.csc_matrix(np.ones((X.shape[0], 1))), X])  



def sigmoid(z):
    return 1 / (1 + np.exp(-z))



def forward(x, w):
    z = np.dot(x, w)
    y_predict = sigmoid(z).reshape((-1, 1)) 
    return y_predict



def predict_accuracy(X_test, Y_test, w, threshold=0.5, label=-1):
    y_predict = forward(X_test, w)

    for i in range(y_predict.shape[0]):
        if y_predict[i, 0] >= threshold:
            y_predict[i, 0] = 1.0
        else:
            y_predict[i, 0] = float(label)

    a = 0 
    for i in range(y_predict.shape[0]):
        if y_predict[i, 0] == Y_test[i]: 
            a += 1
    pred_acc = a / y_predict.shape[0]

    return pred_acc


def sparse_matrix_info(X, len=100):
    temp = X[0:len, :]
    value_list = temp.data



def sparse_matrix_T(X):
    x = sp.find(X)  
    X_T = sp.csr_matrix((x[2], (x[1], x[0])), (X.shape[1], X.shape[0]))
    return X_T



def print_and_save_info(self, opt):

    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = self.parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    # util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def cal_X_var(X):
    X_mean = X.mean()  
    if isinstance(X, np.ndarray):
        X_square = np.multiply(X, X)  
    else:
        X_square = X.multiply(X)
    X_var = X_square.mean() - X_mean ** 2  
    return X_var




def select_dataset(X, Y, m, m_selected):
    assert X.shape[0] == m
    assert len(Y) == m

    np.random.seed(1) 

    index = np.random.choice(m, m_selected, replace=False)    
    index.sort()  
    assert len(index) == m_selected  
    X_selected = X[index, :]
    Y_selected = Y[index]
    return X_selected, Y_selected




def preprocess_sepe(X_train, X_test, save_path=''):
    min_value = min(X_train.min(), X_test.min())
    if (min_value >= 0):
        X_train_new = X_train
        X_test_new = X_test
    else:
        t2 = time.time()
        X_train_new = sp.lil_matrix((X_train.shape[0], X_train.shape[1] * 2), dtype=np.float32)

        X_train = X_train.tocoo()
        non_zero_data = X_train.data 
        non_zero_row = X_train.row
        non_zero_col = X_train.col
        non_zero_num = len(non_zero_data)

        for i in range(non_zero_num):
            row_index = non_zero_row[i] 
            col_index = non_zero_col[i]  
            temp_value = non_zero_data[i]
            if (temp_value < 0):
                X_train_new[row_index, 2 * col_index + 1] = temp_value * (-1)
            else:
                X_train_new[row_index, 2 * col_index] = temp_value
        t3 = time.time()

        X_test_new = sp.lil_matrix((X_test.shape[0], X_test.shape[1] * 2), dtype=np.float32)

        X_test = X_test.tocoo()
        non_zero_data = X_test.data 
        non_zero_row = X_test.row
        non_zero_col = X_test.col
        non_zero_num = len(non_zero_data)

        for i in range(non_zero_num):
            row_index = non_zero_row[i] 
            col_index = non_zero_col[i] 
            temp_value = non_zero_data[i]
            if (temp_value < 0):
                X_test_new[row_index, 2 * col_index + 1] = temp_value * (-1)
            else:
                X_test_new[row_index, 2 * col_index] = temp_value
        t4 = time.time()

        X_train_new = X_train_new.tocsr()
        X_test_new = X_test_new.tocsr()

        if save_path != '':
            train_path = os.path.join(save_path, 'X_train_seperate_norm.npz')
            test_path = os.path.join(save_path, 'X_test_seperate_norm.npz')
            with open(train_path, 'w') as f1:
                sp.save_npz(f1, X_train_new)
                f1.close()
            with open(test_path, 'w') as f2:
                sp.save_npz(f2, X_test_new)
                f2.close()

    return X_train_new, X_test_new



def join_path(opt):
    if opt.sketch_method == 'SM' or opt.sketch_method == 'MM':
        path = os.path.join(opt.main_path, opt.dataset_file_name, 'ratio' + str(int(opt.portion * 10)),
                            opt.sketch_method, 'sketch' + str(opt.k), 'seed' + str(opt.seed))
        
    elif opt.sketch_method == 'RBF' or opt.sketch_method == 'Poly':
        path = os.path.join(opt.main_path, opt.dataset_file_name, 'ratio' + str(int(opt.portion * 10)),
                            opt.sketch_method, 'sketch' + str(opt.k), 'seed' + str(opt.seed), 'gamma' + str(opt.gamma)[:5])

    else:
        print('Kernel method not found!')
        return


    print(f'Path: {path}\n')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def data_preprocess(opt):

    message = '' 
    mes = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '\n'
    message += mes
    print(mes)

    train_data = load_svmlight_file(os.path.join(opt.main_path, opt.dataset_file_name, 'data', opt.train_file_name))
    test_data = load_svmlight_file(os.path.join(opt.main_path, opt.dataset_file_name, 'data', opt.test_file_name))
    X_train = train_data[0].astype(np.float32)
    Y_train = train_data[1].astype(int)
    X_test = test_data[0].astype(np.float32)
    Y_test = test_data[1].astype(int)

  
    if opt.if_select_train == True:
        X_train, Y_train = select_dataset(X_train, Y_train, opt.m_train_raw, opt.m_train)
    if opt.if_select_test == True:
        X_test, Y_test = select_dataset(X_test, Y_test, opt.m_test_raw, opt.m_test)

    if X_train.shape[1] != opt.n:
        X_train = sp.csr_matrix(X_train, shape=(X_train.shape[0], opt.n)) 

    if X_test.shape[1] != opt.n:
        X_test = sp.csr_matrix(X_test, shape=(X_test.shape[0], opt.n))

    if opt.if_normalization == True:
        # save_path = os.path.join(opt.main_path, opt.dataset_file_name, 'data', 'norm')
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        X_train, X_test = preprocess_sepe(X_train, X_test)

    train_class_set = set(Y_train)
    test_class_set = set(Y_test)

    assert len(train_class_set) == opt.class_num

    # preprocess_path = os.path.join(opt.main_path, opt.dataset_file_name)
    # if not os.path.exists(preprocess_path):
    #     os.makedirs(preprocess_path)
    # log_file_name = os.path.join(preprocess_path, 'preprocess_log.txt')
    # with open(log_file_name, 'a') as f:
    #     f.write(message)

    return X_train, X_test, Y_train, Y_test




def Con_samples_to_sketch(m, n, k, b, c, samples):

    assert m == samples.shape[0]  
    assert k == samples.shape[1] 

    if (b == 0 and c == 0):
        X = sp.lil_matrix((m, n * k), dtype=int) 
        sum_zero_sample = 0                       
        for i in range(m):                        
            for j in range(k):
                sample_value = samples[i, j]       
                if (sample_value == -1 or sample_value == 0):  
                    sum_zero_sample += 1
                    continue
                assert sample_value >= 1
                index = n * j + n - sample_value
                X[i, index] = 1
        X = X.tocsr()
        return X


    elif (b == 0 and c != 0):
        X = sp.lil_matrix((m, c * k), dtype=int)
        sum_zero_sample = 0

        np.random.seed(1)
        a_lst = np.random.randint(low=1, high=c, size=k)
        b_lst = np.random.randint(low=0, high=c, size=k)
        for i in range(m):
            for j in range(k):
                sample_value = samples[i, j]
                if (sample_value == -1 or sample_value == 0):
                    sum_zero_sample += 1
                    continue
                assert sample_value >= 1
                hash_value = (a_lst[j] * sample_value + b_lst[j]) % c + 1 
                index = c * j + c - hash_value  
                X[i, index] = 1
        X = X.tocsr()
        return X


    elif (b != 0 and c == 0):
        X = sp.lil_matrix((m, 2 ** b * k), dtype=int)
        sum_zero_sample = 0
        for i in range(m):
            for j in range(k):
                sample_value = samples[i, j]
                if (sample_value == -1 or sample_value == 0):
                    sum_zero_sample += 1
                    continue
                assert sample_value >= 1

                minwise_value = sample_value % (2 ** b) + 1 
                index = 2 ** b * j + 2 ** b - minwise_value 
                X[i, index] = 1
        X = X.tocsr()
        return X
    else:
        print('Wrong Parameters')
        return



def Dis_samples_to_sketch(m, n, n1, k, b, c, samples1, samples2):

    assert m == samples1.shape[0]                     
    assert m == samples2.shape[0]
    assert n1 == samples1.shape[1]
    assert k == samples1.shape[1] + samples2.shape[1]  

    if (b == 0 and c == 0):
        X = sp.lil_matrix((m, n * k), dtype=int)  
        sum_zero_sample1 = 0 
        sum_zero_sample2 = 0  
        sum_zero_instance1 = 0  
        sum_zero_instance2 = 0 
        for i in range(m):  
            if (samples1[i, 0] == -1 or samples1[i, 0] == 0):
                sum_zero_instance1 += 1
            if (samples2[i, 0] == -1 or samples2[i, 0] == 0):
                sum_zero_instance2 += 1
            for j in range(samples1.shape[1]):
                sample_value1 = samples1[i, j]
                if (sample_value1 == -1 or sample_value1 == 0):
                    sum_zero_sample1 += 1
                    continue
                assert sample_value1 >= 1
                index = n * j + n - sample_value1 
                X[i, index] = 1
            for t in range(samples1.shape[1], k):
                sample_value2 = samples2[i, t - samples1.shape[1]]
                if (sample_value2 == -1 or sample_value2 == 0):
                    sum_zero_sample2 += 1
                    continue
                assert sample_value2 >= 1
                index = n * t + n - (sample_value2 + n1)  
                X[i, index] = 1

        X = X.tocsr()
        return X

    elif (b == 0 and c != 0):
        X = sp.lil_matrix((m, c * k), dtype=int)
        sum_zero_sample1 = 0  
        sum_zero_sample2 = 0  
        sum_zero_instance1 = 0 
        sum_zero_instance2 = 0 
        np.random.seed(1)
        a_lst = np.random.randint(low=1, high=c, size=k)
        b_lst = np.random.randint(low=0, high=c, size=k)
        for i in range(m):
            if (samples1[i, 0] == -1 or samples1[i, 0] == 0):
                sum_zero_instance1 += 1
            if (samples2[i, 0] == -1 or samples2[i, 0] == 0):
                sum_zero_instance2 += 1
            for j in range(samples1.shape[1]):
                sample_value1 = samples1[i, j]
                if (sample_value1 == -1 or sample_value1 == 0):
                    sum_zero_sample1 += 1
                    continue
                assert sample_value1 >= 1
                hash_value1 = (a_lst[j] * sample_value1 + b_lst[j]) % c + 1 
                index = c * j + c - hash_value1
                X[i, index] = 1
            for t in range(samples1.shape[1], k):
                sample_value2 = samples2[i, t - samples1.shape[1]]
                if (sample_value2 == -1 or sample_value2 == 0):
                    sum_zero_sample2 += 1
                    continue
                assert sample_value2 >= 1
                hash_value2 = (a_lst[t] * (sample_value2 + n1) + b_lst[t]) % c + 1
                index = c * t + c - hash_value2
                X[i, index] = 1

        X = X.tocsr()
        return X

    elif (b != 0 and c == 0):
        X = sp.lil_matrix((m, 2 ** b * k), dtype=int)
        sum_zero_sample1 = 0  
        sum_zero_sample2 = 0  
        sum_zero_instance1 = 0  
        sum_zero_instance2 = 0  
        for i in range(m):
            if (samples1[i, 0] == -1 or samples1[i, 0] == 0):
                sum_zero_instance1 += 1
            if (samples2[i, 0] == -1 or samples2[i, 0] == 0):
                sum_zero_instance2 += 1
            for j in range(samples1.shape[1]):
                sample_value1 = samples1[i, j]
                if (sample_value1 == -1 or sample_value1 == 0):
                    sum_zero_sample1 += 1
                    continue
                assert sample_value1 >= 1
                minwise_value1 = sample_value1 % (
                            2 ** b) + 1  
                index = 2 ** b * j + 2 ** b - minwise_value1
                X[i, index] = 1
            for t in range(samples1.shape[1], k):
                sample_value2 = samples2[i, t - samples1.shape[1]]
                if (sample_value2 == -1 or sample_value2 == 0):
                    sum_zero_sample2 += 1
                    continue
                assert sample_value2 >= 1
                minwise_value2 = (sample_value2 + n1) % (2 ** b) + 1
                index = 2 ** b * t + 2 ** b - minwise_value2
                X[i, index] = 1

        X = X.tocsr()
        return X

    else:
        print('Wrong parameters')
        return


def Dis_samples_to_sketch2(m, n1, n2, k1, k2, b, c, samples1, samples2):

    assert m == samples1.shape[0]                       
    assert m == samples2.shape[0]
    assert k1 == samples1.shape[1]
    assert k2 == samples2.shape[1]    

    if (b == 0 and c == 0):
        X1 = sp.lil_matrix((m, n1 * k1), dtype=int) 
        X2 = sp.lil_matrix((m, n2 * k2), dtype=int)
        sum_zero_sample1 = 0  
        sum_zero_instance1 = 0  
        sum_zero_sample2 = 0  
        sum_zero_instance2 = 0  
        for i in range(m):   
            if (samples1[i, 0] == -1 or samples1[i, 0] == 0):
                sum_zero_instance1 += 1
            if (samples2[i, 0] == -1 or samples2[i, 0] == 0):
                sum_zero_instance2 += 1
            for j in range(samples1.shape[1]):
                sample_value1 = samples1[i, j]
                if (sample_value1 == -1 or sample_value1 == 0):
                    sum_zero_sample1 += 1
                    continue
                assert sample_value1 >= 1
                index = n1 * j + n1 - sample_value1 
                X1[i, index] = 1
            for t in range(samples2.shape[1]):
                sample_value2 = samples2[i, t]
                if (sample_value2 == -1 or sample_value2 == 0):
                    sum_zero_sample2 += 1
                    continue
                assert sample_value2 >= 1
                index = n2 * t + n2 - sample_value2 
                X2[i, index] = 1

        X1 = X1.tocsr()
        X2 = X2.tocsr()
        X = sp.hstack([X1, X2])
        return X

    elif (b == 0 and c != 0):
        if n1 <= c:
            X1 = sp.lil_matrix((m, n1 * k1), dtype=int)
        else:
            X1 = sp.lil_matrix((m, c * k1), dtype=int)
        if n2 <= c:
            X2 = sp.lil_matrix((m, n2 * k2), dtype=int)
        else:
            X2 = sp.lil_matrix((m, c * k2), dtype=int)
        sum_zero_sample1 = 0 
        sum_zero_sample2 = 0  
        sum_zero_instance1 = 0 
        sum_zero_instance2 = 0 

        np.random.seed(1)
        a_lst = np.random.randint(low=1, high=c, size=k1+k2)
        b_lst = np.random.randint(low=0, high=c, size=k1+k2)
        for i in range(m):
            if (samples1[i, 0] == -1 or samples1[i, 0] == 0):
                sum_zero_instance1 += 1
            if (samples2[i, 0] == -1 or samples2[i, 0] == 0):
                sum_zero_instance2 += 1
            for j in range(k1):
                sample_value1 = samples1[i, j]
                if (sample_value1 == -1 or sample_value1 == 0):
                    sum_zero_sample1 += 1
                    continue
                assert sample_value1 >= 1
                if n1 <= c:
                    index = n1 * j + n1 - sample_value1
                else:
                    hash_value1 = (a_lst[j] * sample_value1 + b_lst[j]) % c + 1  
                    index = c * j + c - hash_value1
                X1[i, index] = 1
            for t in range(k2):
                sample_value2 = samples2[i, t]
                if (sample_value2 == -1 or sample_value2 == 0):
                    sum_zero_sample2 += 1
                    continue
                assert sample_value2 >= 1
                if n2 <= c:
                    index = n2 * t + n2 - sample_value2
                else:
                    hash_value2 = (a_lst[k1+t] * sample_value2 + b_lst[k1+t]) % c + 1
                    index = c * t + c - hash_value2
                X2[i, index] = 1

        X1 = X1.tocsr()
        X2 = X2.tocsr()
        X = sp.hstack([X1, X2])
        return X


    elif (b != 0 and c == 0):
        if n1 <= c:
            X1 = sp.lil_matrix((m, n1 * k1), dtype=int)
        else:
            X1 = sp.lil_matrix((m, 2 ** b * k1), dtype=int)
        if n2 <= c:
            X2 = sp.lil_matrix((m, n2 * k2), dtype=int)
        else:
            X2 = sp.lil_matrix((m, 2 ** b * k2), dtype=int)
        sum_zero_sample1 = 0
        sum_zero_sample2 = 0
        sum_zero_instance1 = 0
        sum_zero_instance2 = 0
        for i in range(m):
            if (samples1[i, 0] == -1 or samples1[i, 0] == 0):
                sum_zero_instance1 += 1
            if (samples2[i, 0] == -1 or samples2[i, 0] == 0):
                sum_zero_instance2 += 1
            for j in range(k1):
                sample_value1 = samples1[i, j]
                if (sample_value1 == -1 or sample_value1 == 0):
                    sum_zero_sample1 += 1
                    continue
                assert sample_value1 >= 1
                minwise_value1 = sample_value1 % (
                            2 ** b) + 1 
                index = 2 ** b * j + 2 ** b - minwise_value1
                X1[i, index] = 1
            for t in range(k2):
                sample_value2 = samples2[i, t]
                if (sample_value2 == -1 or sample_value2 == 0):
                    sum_zero_sample2 += 1
                    continue
                assert sample_value2 >= 1
                minwise_value2 = sample_value2 % (2 ** b) + 1
                index = 2 ** b * t + 2 ** b - minwise_value2
                X2[i, index] = 1
        X1 = X1.tocsr()
        X2 = X2.tocsr()
        X = sp.hstack([X1, X2])
        return X

    else:
        print('Wrong parameters')
        return

