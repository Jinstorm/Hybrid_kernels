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
import WMH_v2
from Option_for_cmd import BaseOptions
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.kernel_approximation import PolynomialCountSketch, Nystroem, RBFSampler



def Con_SamplesGeneration2(opt, X_train, X_test):

    message = ''
    mes = 'Central model sample(sketch) generation for dataset {}. BEGIN...\n'.format(opt.dataset_file_name)
    mes += datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '\n'
    mes += 'Use {} kernel; Total sketch length: {}; Random seed: {}\n'.format(opt.sketch_method, opt.k, opt.seed)
    message += mes
    print(mes)


    if opt.gamma == 0:
        if opt.if_normalization == True:
            gamma = 1. / opt.n / 2
        else:
            gamma = 1. / opt.n
        mes = 'gamma：{}\n'.format(gamma)
        message += mes
        print(mes)
        if opt.gamma_method == 'scale':

            gamma_scale = gamma / cal_X_var(X_train)
            mes = ''
            mes += 'gamma_method：{}\n'.format(opt.gamma_method)
            mes += 'gamma_scale：{}\n'.format(gamma_scale)
            message += mes
            print(mes)
            if opt.sketch_method == 'RBF':
                rff = RBFSampler(gamma=gamma_scale, n_components=opt.k, random_state=opt.seed)
            elif opt.sketch_method == 'Poly':
                ts = PolynomialCountSketch(degree=2, gamma=gamma_scale, coef0=1, n_components=opt.k,
                                           random_state=opt.seed)
        elif opt.gamma_method == 'auto':
            mes = 'gamma_method：{}\n'.format(opt.gamma_method)
            message += mes
            print(mes)

            if opt.sketch_method == 'RBF':
                rff = RBFSampler(gamma=gamma, n_components=opt.k, random_state=opt.seed)
            elif opt.sketch_method == 'Poly':
                ts = PolynomialCountSketch(degree=2, gamma=gamma, coef0=1, n_components=opt.k,
                                           random_state=opt.seed)

    else:
        mes = 'gamma：{}\n'.format(opt.gamma)
        message += mes
        print(mes)
        if opt.sketch_method == 'RBF':
            rff = RBFSampler(gamma=opt.gamma, n_components=opt.k, random_state=opt.seed)
        elif opt.sketch_method == 'Poly':
            ts = PolynomialCountSketch(degree=2, gamma=opt.gamma, coef0=1, n_components=opt.k,
                                       random_state=opt.seed)
    if opt.sketch_method == 'RBF':
        train_samples = rff.fit_transform(X_train)
        test_samples = rff.fit_transform(X_test)
    elif opt.sketch_method == 'Poly':
        train_samples = ts.fit_transform(X_train)
        test_samples = ts.fit_transform(X_test)
    else:
        print('Wrong kernel method!')
        return

    mes = 'The shape of generated sample(sketch): train {}; test {}\n'.format(train_samples.shape, test_samples.shape)
    message += mes
    print(mes)

    X_samples_path = join_path(opt)
    if opt.save_file:
        
        mes = 'Path: {}\n'.format(X_samples_path)
        message += mes
        print(mes)
        X_train_samples_file = os.path.join(X_samples_path, 'X_train_samples.txt')
        X_test_samples_file = os.path.join(X_samples_path, 'X_test_samples.txt')
        np.savetxt(X_train_samples_file, train_samples, delimiter=',')
        np.savetxt(X_test_samples_file, test_samples, delimiter=',')

    log_file_name = os.path.join(X_samples_path, 'samples_generation_log.txt')

    mes = 'Central model sample(sketch) generation for dataset {}. END.\n\n\n'.format(opt.dataset_file_name)
    message += mes
    print(mes)

    with open(log_file_name, 'a') as f:
        f.write(message)

    return train_samples, test_samples

def Dis_SamplesGeneration2(opt, X_train, X_test):

    message = ''
    mes = ''
    mes += 'Distributed model sample(sketch) generation for dataset {}. Random seed {}. Feature partition ratio {}:{}. BEGIN...\n'.format(opt.dataset_file_name, opt.seed, int(opt.portion * 10),  10 - int(opt.portion * 10))
    mes += datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '\n'
    mes += 'Use {} kernel; Total sketch length: {}\n'.format(opt.sketch_method, opt.k)
    message += mes
    print(mes)

    if opt.if_normalization == True:
        n1 = np.floor(2 * opt.n * opt.portion).astype(int)
        n2 = 2 * opt.n - n1 
    else:
        n1 = np.floor(opt.n * opt.portion).astype(int) 
        n2 = opt.n - n1 

    k1 = np.floor(opt.k * opt.portion).astype(int) 
    k2 = opt.k - k1 

    X1_train, X2_train = util_feature2(X_train, n1)
    X1_test, X2_test = util_feature2(X_test, n1)

    if opt.gamma == 0:
        if opt.if_normalization == True:
            gamma = 1. / opt.n / 2
        else:
            gamma = 1. / opt.n
        gamma1 = 1. / n1
        gamma2 = 1. / n2

        mes = 'gamma: {}\n'.format(gamma)
        message += mes
        print(mes)
        if opt.gamma_method == 'scale':

            gamma_scale1 = gamma1 / cal_X_var(X1_train)
            gamma_scale2 = gamma2 / cal_X_var(X2_train)
            gamma_scale = gamma / cal_X_var(X_train)
            mes = ''
            mes += 'gamma_method: {}\n'.format(opt.gamma_method)
            mes += 'gamma_scale: {}\n'.format(gamma_scale)
            message += mes
            print(mes)

            if opt.sketch_method == 'RBF':
                rff1 = RBFSampler(gamma=gamma_scale, n_components=k1, random_state=opt.seed)
                rff2 = RBFSampler(gamma=gamma_scale, n_components=k2, random_state=opt.seed)
            elif opt.sketch_method == 'Poly':
                ts1 = PolynomialCountSketch(degree=2, gamma=gamma_scale, coef0=1, n_components=k1,
                                            random_state=opt.seed)
                ts2 = PolynomialCountSketch(degree=2, gamma=gamma_scale, coef0=1, n_components=k2,
                                            random_state=opt.seed)

        elif opt.gamma_method == 'auto':
            mes = 'gamma_method：{}\n'.format(opt.gamma_method)
            message += mes
            print(mes)

            if opt.sketch_method == 'RBF':
                rff1 = RBFSampler(gamma=gamma, n_components=k1, random_state=opt.seed)
                rff2 = RBFSampler(gamma=gamma, n_components=k2, random_state=opt.seed)
            elif opt.sketch_method == 'Poly':
                ts1 = PolynomialCountSketch(degree=2, gamma=gamma, coef0=1, n_components=k1, random_state=opt.seed)
                ts2 = PolynomialCountSketch(degree=2, gamma=gamma, coef0=1, n_components=k2, random_state=opt.seed)


    else:
        mes = 'gamma: {}\n'.format(opt.gamma)
        message += mes
        print(mes)
        if opt.sketch_method == 'RBF':
            rff1 = RBFSampler(gamma=opt.gamma, n_components=k1, random_state=opt.seed)
            rff2 = RBFSampler(gamma=opt.gamma, n_components=k2, random_state=opt.seed)
        elif opt.sketch_method == 'Poly':
            ts1 = PolynomialCountSketch(degree=2, gamma=opt.gamma, coef0=1, n_components=k1, random_state=opt.seed)
            ts2 = PolynomialCountSketch(degree=2, gamma=opt.gamma, coef0=1, n_components=k2, random_state=opt.seed)

    if opt.sketch_method == 'RBF':
        train_samples1 = rff1.fit_transform(X1_train)
        test_samples1 = rff1.fit_transform(X1_test)
        train_samples2 = rff2.fit_transform(X2_train)
        test_samples2 = rff2.fit_transform(X2_test)
    elif opt.sketch_method == 'Poly':
        train_samples1 = ts1.fit_transform(X1_train)
        test_samples1 = ts1.fit_transform(X1_test)
        train_samples2 = ts2.fit_transform(X2_train)
        test_samples2 = ts2.fit_transform(X2_test)
    else: 
        print('Wrong kernel method!')
        return

    mes = 'The shape of generated sample(sketch): train1 {}; test1 {}; train2 {}; test2 {}\n'.format(train_samples1.shape, test_samples1.shape, train_samples2.shape, test_samples2.shape)
    message += mes
    print(mes)

    X_samples_path = join_path(opt)
    if opt.save_file:
        mes = 'Path: {}\n'.format(X_samples_path)
        message += mes
        print(mes)
        X1_train_samples_file = os.path.join(X_samples_path, 'X1_train_samples.txt')
        X1_test_samples_file = os.path.join(X_samples_path, 'X1_test_samples.txt')
        X2_train_samples_file = os.path.join(X_samples_path, 'X2_train_samples.txt')
        X2_test_samples_file = os.path.join(X_samples_path, 'X2_test_samples.txt')
        np.savetxt(X1_train_samples_file, train_samples1, delimiter=',')
        np.savetxt(X1_test_samples_file, test_samples1, delimiter=',')
        np.savetxt(X2_train_samples_file, train_samples2, delimiter=',')
        np.savetxt(X2_test_samples_file, test_samples2, delimiter=',')

    log_file_name = os.path.join(X_samples_path, 'samples_generation_log.txt')
    mes += 'Distributed model sample(sketch) generation for dataset {}. END.\n\n\n'.format(opt.dataset_file_name)
    message += mes
    print(mes)

    with open(log_file_name, 'a') as f:
        f.write(message)

    return train_samples1, test_samples1, train_samples2, test_samples2









