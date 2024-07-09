from liblinear.liblinearutil import *
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
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
from sklearn.kernel_approximation import PolynomialCountSketch, Nystroem, RBFSampler
from SamplesGeneration1 import Con_SamplesGeneration1, Dis_SamplesGeneration1
from SamplesGeneration2 import Con_SamplesGeneration2, Dis_SamplesGeneration2


def DistributedModel(opt, X_train, X_test, Y_train, Y_test):

    message = ''
    mes = ''
    mes += 'Distributed model training and testing experiment for dataset {}. Random seed {}. Feature partition ratio {}:{}. BEGIN...\n'.format(opt.dataset_file_name, opt.seed, int(opt.portion * 10),  10 - int(opt.portion * 10))
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


    # partition
    if opt.if_normalization == True:
        n1 = np.floor(2 * opt.n * opt.portion).astype(int) 
        n2 = 2 * opt.n - n1
    else:
        n1 = np.floor(opt.n * opt.portion).astype(int)
        n2 = opt.n - n1
    X1_train, X2_train = util_feature2(X_train, n1)
    X1_test, X2_test = util_feature2(X_test, n1)

    k1 = np.floor(opt.k * opt.portion).astype(int)
    k2 = opt.k - k1

    mes = 'Sketch length for each partition: {}, {}\n'.format(k1, k2)
    message += mes
    print(mes)


    # generate sketch
    if opt.sketch_method == 'SM' or opt.sketch_method == 'MM':

        # # load from files
        # X1_train_samples_file = os.path.join(X_samples_path, 'X1_train_samples.txt')
        # X1_test_samples_file = os.path.join(X_samples_path, 'X1_test_samples.txt')
        # X2_train_samples_file = os.path.join(X_samples_path, 'X2_train_samples.txt')
        # X2_test_samples_file = os.path.join(X_samples_path, 'X2_test_samples.txt')
        # train1_samples = np.loadtxt(X1_train_samples_file, delimiter=',', dtype='int')
        # test1_samples = np.loadtxt(X1_test_samples_file, delimiter=',', dtype='int')
        # train2_samples = np.loadtxt(X2_train_samples_file, delimiter=',', dtype='int')
        # test2_samples = np.loadtxt(X2_test_samples_file, delimiter=',', dtype='int')

        train1_samples, test1_samples, train2_samples, test2_samples = Dis_SamplesGeneration1(opt, X_train, X_test)

        X_train_sketch = Dis_samples_to_sketch2(opt.m_train, n1, n2, k1, k2, opt.b, opt.c, train1_samples, train2_samples)
        X_test_sketch = Dis_samples_to_sketch2(opt.m_test, n1, n2, k1, k2, opt.b, opt.c, test1_samples, test2_samples)


    elif opt.sketch_method == 'RBF' or opt.sketch_method == 'Poly':

        X1_train_sketch, X1_test_sketch, X2_train_sketch, X2_test_sketch = Dis_SamplesGeneration2(opt, X_train, X_test)

        X_train_sketch = np.concatenate((X1_train_sketch, X2_train_sketch), axis=1)
        X_test_sketch = np.concatenate((X1_test_sketch, X2_test_sketch), axis=1)

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
    mes += 'Distributed model training and testing experiment for dataset {}. Random seed {}. Feature partition ratio {}:{}. END.\n\n\n'.format(opt.dataset_file_name, opt.seed, int(opt.portion * 10),  10 - int(opt.portion * 10))
    message += mes
    print(mes)

    log_file_name = os.path.join(experiment_path, 'experiment_log.txt')
    with open(log_file_name, 'a') as f:
        f.write(message)

