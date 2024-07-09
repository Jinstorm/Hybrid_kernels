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


def Con_SamplesGeneration1(opt, X_train, X_test):

    message = ''
    mes = 'Central model sample(sketch) generation for dataset {}. BEGIN...\n'.format(opt.dataset_file_name)
    mes += datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '\n'
    mes += 'Use {} kernel; Total sketch length: {}; Random seed: {}\n'.format(opt.sketch_method, opt.k, opt.seed)
    message += mes
    print(mes)

    wmh_train = WMH_v2.WeightedMinHash(sp.csr_matrix(X_train), opt.k, seed=opt.seed)
    if opt.sketch_method == 'SM':
        train_samples, train_elapsed, zero_num = wmh_train.P_minhash()

    elif opt.sketch_method == 'MM':
        train_samples, train_elapsed, zero_num = wmh_train.licws()               

    else:
        print('Wrong kernel method!')
        return


    wmh_test = WMH_v2.WeightedMinHash(sp.csr_matrix(X_test), opt.k, seed=opt.seed)
    if opt.sketch_method == 'SM':
        test_samples, test_elapsed, zero_num = wmh_test.P_minhash()

    elif opt.sketch_method == 'MM':
        test_samples, test_elapsed, zero_num = wmh_test.licws()                            

    else:
        print('Wrong kernel method!')
        return


    train_samples = train_samples.astype(int)
    test_samples = test_samples.astype(int)
    train_samples = train_samples + np.ones(train_samples.shape, dtype=int)
    test_samples = test_samples + np.ones(test_samples.shape, dtype=int)


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
        np.savetxt(X_train_samples_file, train_samples, fmt='%i', delimiter=',') 
        np.savetxt(X_test_samples_file, test_samples, fmt='%i', delimiter=',')

    log_file_name = os.path.join(X_samples_path, 'samples_generation_log.txt')
    mes = 'Central model sample(sketch) generation for dataset {}. END.\n\n\n'.format(opt.dataset_file_name)
    message += mes
    print(mes)

    with open(log_file_name, 'a') as f:
        f.write(message)

    return train_samples, test_samples

def Dis_SamplesGeneration1(opt, X_train, X_test):


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

    wmh_train1 = WMH_v2.WeightedMinHash(X1_train, k1, seed=opt.seed)
    wmh_train2 = WMH_v2.WeightedMinHash(X2_train, k2, seed=opt.seed)
    if opt.sketch_method == 'SM':
        train_samples1, train_elapsed1, zero_num1 = wmh_train1.P_minhash()
        train_samples2, train_elapsed2, zero_num2 = wmh_train2.P_minhash()
    elif opt.sketch_method == 'MM':
        train_samples1, train_elapsed1, zero_num1 = wmh_train1.licws()       
        train_samples2, train_elapsed2, zero_num2 = wmh_train2.licws()

    else:
        print('Wrong kernel method!')
        return


    wmh_test1 = WMH_v2.WeightedMinHash(X1_test, k1, seed=opt.seed)
    wmh_test2 = WMH_v2.WeightedMinHash(X2_test, k2, seed=opt.seed)
    if opt.sketch_method == 'SM':
        test_samples1, test_elapsed1, zero_num1 = wmh_test1.P_minhash()
        test_samples2, test_elapsed2, zero_num2 = wmh_test2.P_minhash()

    elif opt.sketch_method == 'MM':
        test_samples1, test_elapsed1, zero_num1 = wmh_test1.licws()          
        test_samples2, test_elapsed2, zero_num2 = wmh_test2.licws()

    else:
        print('Wrong kernel method!')
        return

    train_samples1 = train_samples1.astype(int)
    test_samples1 = test_samples1.astype(int)
    train_samples1 = train_samples1 + np.ones(train_samples1.shape, dtype=int)
    test_samples1 = test_samples1 + np.ones(test_samples1.shape, dtype=int)
    train_samples2 = train_samples2.astype(int)
    test_samples2 = test_samples2.astype(int)
    train_samples2 = train_samples2 + np.ones(train_samples2.shape, dtype=int)
    test_samples2 = test_samples2 + np.ones(test_samples2.shape, dtype=int)

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
        np.savetxt(X1_train_samples_file, train_samples1, fmt='%i', delimiter=',')
        np.savetxt(X1_test_samples_file, test_samples1, fmt='%i', delimiter=',')
        np.savetxt(X2_train_samples_file, train_samples2, fmt='%i', delimiter=',')
        np.savetxt(X2_test_samples_file, test_samples2, fmt='%i', delimiter=',')

    log_file_name = os.path.join(X_samples_path, 'samples_generation_log.txt')
    mes += 'Distributed model sample(sketch) generation for dataset {}. END.\n\n\n'.format(opt.dataset_file_name)
    message += mes
    print(mes)

    
    with open(log_file_name, 'a') as f:
        f.write(message)

    return train_samples1, test_samples1, train_samples2, test_samples2












