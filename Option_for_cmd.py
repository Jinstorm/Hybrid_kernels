import argparse
import os
import numpy as np
import datetime



class BaseOptions():
    def __init__(self):
        parser = argparse.ArgumentParser("Model Parameters settings")

        parser.add_argument('--main_path', type=str, default='./dataset',
                            help='main path of the entire dataset, contains raw and samples file')
        parser.add_argument('--dataset_file_name', type=str, default='kits',
                            help='the name of dataset, also the file nameof dataset') 
        parser.add_argument('--train_file_name', type=str, default='', help='')
        parser.add_argument('--test_file_name', type=str, default='', help='')
        parser.add_argument('--if_normalization', type=bool, default=False, help='if normalize the raw dataset or not')
        parser.add_argument('--if_select_train', type=bool, default=False, help='if select train dataset or not')
        parser.add_argument('--if_select_test', type=bool, default=False, help='if select test dataset or not')

        parser.add_argument('--save_file', action='store_true',
                            help='if need to save sample(sketch) files')
        parser.add_argument('--portion', type=float, default=0.5,
                            help='the portion of distributed model, if portion=0, use the concentrated model')
        parser.add_argument('--sketch_method', type=str, default='SM',
                            help='MM, SM, RBF, Poly')
        parser.add_argument('--k', type=int, default=1024,
                            help='times of consistent weigthed sampling or the times of sampling in RFF')
        parser.add_argument('--seed', type=int, default=1, help='random seed')

        parser.add_argument('--C', type=float, default=1.0,
                            help='regularization parameter of LIBLINEAR model')
        parser.add_argument('--param', type=str, default='-s 0 -e 0.0001 -q',
                            help='other parameters of LIBLINEAR model')
        parser.add_argument('--b', type=int, default=0, help='b bit minwise hash, if b=0, do not use minwise hash')  # @
        parser.add_argument('--c', type=int, default=0, help='Count Sketch, if c=0, do not use Count Sketch')  # @
        parser.add_argument('--gamma', type=float, default=0,
                            help='the parameter of RBF Kernel, if gamma=0, use the default method for gamma calculation: 1./n_features, gamma_scale = gamma/X.var()')
        parser.add_argument('--gamma_method', type=str, default='scale',
                            help='the method for calculating gamma in RBF Kernel used in sklearn.svm.SVC, for example, scale, auto')

        parser.add_argument('--m_train', type=int, default=0, help='')
        parser.add_argument('--m_test', type=int, default=0, help='')
        parser.add_argument('--m_train_raw', type=int, default=0, help='')
        parser.add_argument('--m_test_raw', type=int, default=0, help='')
        parser.add_argument('--n', type=int, default=0, help='')
        parser.add_argument('--class_num', type=int, default=2, help='the number of dataset classes')
        self.parser = parser

    def parse(self):
        opt = self.parser.parse_args()
        if opt.dataset_file_name == 'pendigits':
            self.parser.set_defaults(train_file_name='pendigits.txt', test_file_name='pendigits.t', m_train=7494,
                                     m_test=3498, m_train_raw=7494, m_test_raw=3498, n=16, class_num=10)

        elif opt.dataset_file_name == 'letter':
            self.parser.set_defaults(train_file_name='letter.scale.txt', test_file_name='letter.scale.t.txt',
                                     m_train=15000,
                                     m_test=5000, m_train_raw=15000, m_test_raw=5000, n=16, class_num=26,
                                     if_normalization=True)

        elif opt.dataset_file_name == 'segment':
            self.parser.set_defaults(train_file_name='segment_train.txt', test_file_name='segment_test.txt',
                                     m_train=1155,
                                     m_test=1155, m_train_raw=1155, m_test_raw=1155, n=19, class_num=7,
                                     if_normalization=True)

        elif opt.dataset_file_name == 'ijcnn':
            self.parser.set_defaults(train_file_name='ijcnn1.bz2',
                                     test_file_name='ijcnn1.t.bz2',
                                     m_train=10000,
                                     m_test=10000, m_train_raw=49990, m_test_raw=91701, n=22,
                                     class_num=2, if_normalization=True, if_select_train=True, if_select_test=True)

        elif opt.dataset_file_name == 'satimage':
            self.parser.set_defaults(train_file_name='satimage.scale.txt', test_file_name='satimage.scale.t',
                                     m_train=4435,
                                     m_test=2000, m_train_raw=4435, m_test_raw=2000, n=36, class_num=6,
                                     if_normalization=True)

        elif opt.dataset_file_name == 'sensorless':
            self.parser.set_defaults(train_file_name='Sensorless.scale.tr',
                                     test_file_name='Sensorless.scale.val.txt',
                                     m_train=48509,
                                     m_test=10000, m_train_raw=48509, m_test_raw=10000, n=48,
                                     class_num=11)

        elif opt.dataset_file_name == 'covertype':
            self.parser.set_defaults(train_file_name='covertype_train.txt',
                                     test_file_name='covertype_test.txt',
                                     m_train=10000,
                                     m_test=10000, m_train_raw=10000, m_test_raw=10000, n=54,
                                     class_num=7)

        elif opt.dataset_file_name == 'splice':
            self.parser.set_defaults(train_file_name='splice_train.txt', test_file_name='splice_test', m_train=1000,
                                     m_test=2175, m_train_raw=1000, m_test_raw=2175, n=60, class_num=2)

        elif opt.dataset_file_name == 'SensIT':
            self.parser.set_defaults(train_file_name='combined.bz2',
                                     test_file_name='combined.t.bz2',
                                     m_train=10000,
                                     m_test=19705, m_train_raw=78823, m_test_raw=19705, n=100,
                                     class_num=3, if_normalization=True, if_select_train=True)

        elif opt.dataset_file_name == 'usps':
            self.parser.set_defaults(train_file_name='usps', test_file_name='usps.t', m_train=7291,
                                     m_test=2007, m_train_raw=7291, m_test_raw=2007, n=256, class_num=10,
                                     if_normalization=True)

        elif opt.dataset_file_name == 'mnist':
            self.parser.set_defaults(train_file_name='mnist', test_file_name='mnist.t', m_train=60000,
                                     m_test=10000, m_train_raw=60000, m_test_raw=10000, n=780, class_num=10)

        elif opt.dataset_file_name == 'YoutubeAudio':
            self.parser.set_defaults(train_file_name='YoutubeAudio_train.txt', test_file_name='YoutubeAudio_test.txt',
                                     m_train=10000,
                                     m_test=11930, m_train_raw=10000, m_test_raw=11930, n=2000,
                                     class_num=31)
                                
        elif opt.dataset_file_name == 'dilbert':
            self.parser.set_defaults(train_file_name='dilbert_train.txt',
                                     test_file_name='dilbert_test.txt',
                                     m_train=7000,
                                     m_test=3000, m_train_raw=7000, m_test_raw=3000, n=2000,
                                     class_num=5)

        elif opt.dataset_file_name == 'SEMG1':
            self.parser.set_defaults(train_file_name='SEMG1_train.txt', test_file_name='SEMG1_test.txt',
                                     m_train=900,
                                     m_test=900, m_train_raw=900, m_test_raw=900, n=3000,
                                     class_num=12, if_normalization=True)

        elif opt.dataset_file_name == 'cifar10':
            self.parser.set_defaults(train_file_name='cifar10_train.txt', test_file_name='cifar10_test.txt',
                                     m_train=10000,
                                     m_test=10000, m_train_raw=50000, m_test_raw=10000, n=3072,
                                     class_num=10, if_select_train=True)
                                     
        elif opt.dataset_file_name == 'DailySports':
            self.parser.set_defaults(train_file_name='DailySports_train.txt', test_file_name='DailySports_test.txt',
                                     m_train=4560,
                                     m_test=4560, m_train_raw=4560, m_test_raw=4560, n=5625,
                                     class_num=19, if_normalization=True)

        elif opt.dataset_file_name == 'kits':
            self.parser.set_defaults(train_file_name='kits_train.txt', test_file_name='kits_test.txt',
                                     m_train=700,
                                     m_test=300, m_train_raw=700, m_test_raw=300, n=27648,
                                     class_num=2)

        elif opt.dataset_file_name == 'webspam10k':
            self.parser.set_defaults(train_file_name='webspam10k_train.txt', test_file_name='webspam10k_test.txt',
                                     m_train=10000,
                                     m_test=10000, m_train_raw=10000, m_test_raw=10000, n=8355099,
                                     class_num=2)

        else:
            print('Note that dataset {} has not been stored. Please manually set the parameters for the dataset.'.format(opt.dataset_file_name))

        return self.parser.parse_args()


    def print_options(self, opt, log_file_name):
        message = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '\n'
        for k, v in vars(opt).items():
            message += '{} = {}\n'.format(str(k), str(v))
        print(message)

        with open(log_file_name, 'a') as f:
            f.write('Model parameters settings\n')
            f.write(message)

