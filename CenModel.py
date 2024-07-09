from liblinear.liblinearutil import *
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.datasets import load_svmlight_file
import time
import os
from utils import *
import datetime
from sklearn.preprocessing import StandardScaler
from Option_for_cmd import BaseOptions
from sklearn.kernel_approximation import PolynomialCountSketch, RBFSampler
from SamplesGeneration1 import Con_SamplesGeneration1, Dis_SamplesGeneration1
from SamplesGeneration2 import Con_SamplesGeneration2, Dis_SamplesGeneration2


def CentralModel(opt, X_train, X_test, Y_train, Y_test):

    message = ''
    mes = ''
    mes += 'Central model training and testing experiment for dataset {}. Random seed {}. BEGIN...\n'.format(opt.dataset_file_name, opt.seed)
    mes += datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '\n'
    message += mes
    print(mes)

    mes = ''
    mes += 'Use {} kernel\n'.format(opt.sketch_method)
    mes += 'Total sketch length: {}\n'.format(opt.k)
    # mes += 'b bit minwise hash：{}\n'.format(opt.b)
    mes += 'Compression factors: {}\n'.format(opt.c)
    message += mes
    print(mes)

    # generate sketch
    # X_samples_path = join_path(opt)

    if opt.sketch_method == 'SM' or opt.sketch_method == 'MM':
        # load from files
        # X_train_samples_file = os.path.join(X_samples_path, 'X_train_samples.txt')
        # X_test_samples_file = os.path.join(X_samples_path, 'X_test_samples.txt')
        # train_samples = np.loadtxt(X_train_samples_file, delimiter=',', dtype='int')
        # test_samples = np.loadtxt(X_test_samples_file, delimiter=',', dtype='int')

        train_samples, test_samples = Con_SamplesGeneration1(opt, X_train, X_test)

        if opt.if_normalization == True:
            X_train_sketch = Con_samples_to_sketch(opt.m_train, opt.n * 2, opt.k, opt.b, opt.c, train_samples)
            X_test_sketch = Con_samples_to_sketch(opt.m_test, opt.n * 2, opt.k, opt.b, opt.c, test_samples)
        else:
            X_train_sketch = Con_samples_to_sketch(opt.m_train, opt.n, opt.k, opt.b, opt.c, train_samples)
            X_test_sketch = Con_samples_to_sketch(opt.m_test, opt.n, opt.k, opt.b, opt.c, test_samples)


    elif opt.sketch_method == 'RBF' or opt.sketch_method == 'Poly':
        
        X_train_sketch, X_test_sketch = Con_SamplesGeneration2(opt, X_train, X_test)
        
    else:
        print('Wrong kernel method!')
        return

    # LIBLINEAR training and testing

    mes = ''
    mes += 'Regularization parameter C of LIBLINEAR model: -c {}\n'.format(opt.C)
    mes += 'Other parameters of LIBLINEAR model: {}\n'.format(opt.param)
    message += mes
    print(mes)

    param = opt.param + ' -c ' + str(opt.C)

    t0 = time.time()
    m = train(Y_train, X_train_sketch, param)
    train_time = time.time() - t0

    t1 = time.time()
    p_labs, p_acc, p_vals = predict(Y_test, X_test_sketch, m)
    test_time = time.time() - t1

    mes = ''
    mes += 'The experiment of the hybrid kernels with {} kernel method\n'.format(opt.sketch_method)
    mes += 'Accuracy: {:.3f}%\n'.format(p_acc[0])
    message += mes
    print(mes)


    experiment_path = join_path(opt)
    mes = 'Experiment log path: {}\n'.format(experiment_path)
    mes += 'Central model training and testing experiment for dataset {}. Random seed {}. END.\n\n\n'.format(opt.dataset_file_name, opt.seed)
    message += mes
    print(mes)

    log_file_name = os.path.join(experiment_path, 'experiment_log.txt')
    with open(log_file_name, 'a') as f:
        f.write(message)



