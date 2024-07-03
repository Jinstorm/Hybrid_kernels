import os
import sys
import time
import math
import datetime
import argparse
import numpy as np
from multiprocessing.pool import Pool
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_pardir = os.path.join(dir_path, os.pardir)
abs_parpardir = os.path.join(abs_pardir, os.pardir)
sys.path.append(abs_pardir)
sys.path.append(abs_parpardir)
from secureModel import SecureLRModel

from paillierm.encrypt import PaillierEncrypt
from paillierm.fixedpoint import FixedPointEndec
from paillierm.utils import urand_tensor

""" data processing and loading method """
from data_processing.dataset_select import dataset_selector
from data_processing.kernel_approximate_method import dataSketch_generator_RBFandPolyKernel

PATH_DATA = '../../data/'

class LogisticRegression(SecureLRModel):
    """
    secure logistic regression model
    """
    def __init__(self, weight_vector, batch_size, max_iter, alpha, 
                        eps, ratio = None, penalty = None, lambda_para = 1, data_tag = None, ovr = None, sigmoid_func = None,
                        sketch_tag = None, countsketch_c = 0, bbitmwhash_b = 0, dataset_name = None, 
                        kernel_method = None, sampling_k = None, Epoch_list_max = None, logger = None, scalering_raw = None, c_penalty = 1):
        super(LogisticRegression, self).__init__("CAESAR")
        self.model_weights = weight_vector
        self.batch_size = batch_size
        self.batch_num = []
        self.n_iteration = 0
        self.max_iter = max_iter
        self.alpha = alpha
        self.pre_loss = 0
        self.eps = eps
        self.ratio = ratio
        self.penalty = penalty
        self.lambda_para = lambda_para
        self.data_tag = data_tag # sparse/dense
        self.ovr = ovr
        self.countsketch_c = countsketch_c  # countsketch-c
        self.bbitmwhash_b = bbitmwhash_b # bbitminwisehash-b
        self.kernel_method = kernel_method
        self.sampling_k = sampling_k
        self.sigmoid_func = sigmoid_func
        self.scalering_raw = scalering_raw
        self.c_penalty = c_penalty

        self.logger = logger
        self.dataset_name = dataset_name
        self.sketch_tag = sketch_tag
        self.training_status = "normal"

        # WAN(Wide area network) Bandwidth
        self.WAN_bandwidth = 10 # Mbps
        self.train_commTime_account = 0
        self.inference_compute_time = 0
        self.inference_time_account = 0
        self.mem_occupancy = 8 

        ## compute time
        self.offline_calculate_time = 0

        # enc init
        self.cipher = PaillierEncrypt()
        self.cipher.generate_key()
        self.fixedpoint_encoder = FixedPointEndec(n = 1e10)

        # Epoch
        EPOCH_list = []
        EPOCH_list = [i for i in range(1, Epoch_list_max + 1)]

        self.EPOCH_list = EPOCH_list # eg. [1,5,10,15,20,25,30,35,40]
        assert(self.EPOCH_list[-1] <= self.max_iter)
        
        if self.ovr == "bin": self.modelWeight_and_Time_List = dict()
        elif self.ovr == "ovr": self.OVRModel_Agg = dict()

        filename = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f')
        if sketch_tag == "sketch":
            if kernel_method == "pminhash": self.logname = "CAESAR_" + dataset_name + "_" + kernel_method + sampling_k + "_" + str(countsketch_c) + "_" + filename + ".txt"
            elif kernel_method == "0bitcws": self.logname = "CAESAR_" + dataset_name + "_" + kernel_method + sampling_k + "_" + str(countsketch_c) + "_" + filename + ".txt"
            else: self.logname = "CAESAR_" + dataset_name + "_" + kernel_method + sampling_k + "_" + filename + ".txt"
        else:
            self.logname = "CAESAR_" + dataset_name + "_raw_" + filename + ".txt"

        self.time_start_outer = 0
        dirName = None
        if sketch_tag == "sketch": dirName = kernel_method
        else: dirName = sketch_tag
        fileAbsPath = os.path.join(os.getcwd(), dataset_name, dirName, self.logname)
        try:
            File_Path = os.path.join(os.getcwd(), dataset_name, dirName)
            if not os.path.exists(File_Path):
                os.makedirs(File_Path)
            self.logname = fileAbsPath

        except:
            raise FileNotFoundError("Logfile Path not exits.")


    def _cal_z(self, weights, features, party = None, encrypt = None):
        if encrypt is not None:
            if party == "A": self.za1 = np.dot(features, weights.T)
            elif party == "B": self.zb2 = np.dot(features, weights.T)
            else: raise NotImplementedError
        elif party == "A":
            if self.data_tag == 'sparse': self.wx_self_A = features.dot(weights.T)
            else: self.wx_self_A = np.dot(features, weights.T)
        elif party == "B": 
            if self.data_tag == 'sparse': self.wx_self_B = features.dot(weights.T)
            else: self.wx_self_B = np.dot(features, weights.T)
        else: 
            if self.data_tag == 'sparse':
                self.wx_self = features.dot(weights.T)
            elif self.data_tag == None:
                self.wx_self = np.dot(features, weights.T)

    def _compute_sigmoid(self, z, z_cube = None):
        if self.sigmoid_func == "linear":
            return z * 0.25 + 0.5
        elif self.sigmoid_func == "cube":
            return 0.5 + 0.197 * z - 0.004 * z_cube
        else:
            raise NotImplementedError("Invalid sigmoid approximation.")

    def _compute_sigmoid_dual_distributed(self, z):
        return z * 0.25

    def distributed_compute_loss_cross_entropy(self, label, batch_num):
        """
            Use Taylor series expand log loss:
            Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
            Then loss' = - (1/N)*∑(log(1/2) - 1/2*wx + ywx -1/8(wx)^2)
        """
        self.encrypted_wx = self.wx_self_A + self.wx_self_B
        half_wx = -0.5 * self.encrypted_wx
        ywx = self.encrypted_wx * label
        wx_square = (2*self.wx_self_A * self.wx_self_B + self.wx_self_A * self.wx_self_A + self.wx_self_B * self.wx_self_B) * -0.125
        loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) - np.log(0.5) )
        return loss

    def forward(self, weights, features):
        self._cal_z(weights, features, party = None, encrypt = None)
        sigmoid_z = self._compute_sigmoid(self.wx_self)
        return sigmoid_z

    def distributed_forward(self, weights, features, party = None):
        self._cal_z(weights, features, party, encrypt = None)
        if party == "A": sigmoid_z = self._compute_sigmoid(self.wx_self_A)
        elif party == "B": sigmoid_z = self._compute_sigmoid_dual_distributed(self.wx_self_B)
        return sigmoid_z

    def distributed_backward(self, error, features, batch_num):
        if self.data_tag == 'sparse': gradient = features.T.dot(error).T / batch_num
        elif self.data_tag == None: gradient = np.dot(error.T, features) / batch_num
        return gradient

    def check_converge_by_loss(self, loss):
        converge_flag = False
        if self.pre_loss is None: pass
        elif abs(self.pre_loss - loss) < self.eps: converge_flag = True
        self.pre_loss = loss
        return converge_flag
    
    def shuffle_distributed_data(self, XdatalistA, XdatalistB, Ydatalist):
        zip_list = list( zip(XdatalistA, XdatalistB, Ydatalist) )
        np.random.shuffle(zip_list)
        XdatalistA[:], XdatalistB[:], Ydatalist[:] = zip(*zip_list)
        return XdatalistA, XdatalistB, Ydatalist


    def time_counting(self, tensor):
        if tensor.ndim == 2: object_num = tensor.shape[0] * tensor.shape[1]
        else: object_num = tensor.shape[0]
        commTime = object_num * self.mem_occupancy / (1024*1024) / (self.WAN_bandwidth/8)
        self.train_commTime_account += commTime

    def time_counting_model_inference(self, tensor):
        if tensor.ndim == 2: object_num = tensor.shape[0] * tensor.shape[1]
        else: object_num = tensor.shape[0]
        commTime = object_num * self.mem_occupancy / (1024*1024) / (self.WAN_bandwidth/8)
        self.inference_time_account += commTime

    def secret_share_vector_plaintext(self, share_target, flag = None):
        _pre = urand_tensor(q_field = self.fixedpoint_encoder.n, tensor = share_target)
        tmp = self.fixedpoint_encoder.decode(_pre)
        share = share_target - tmp
        if flag == "inference": self.time_counting_model_inference(share)
        else: self.time_counting(share)
        return tmp, share

    def secret_share_vector(self, share_target, flag = None):
        _pre = urand_tensor(q_field = self.fixedpoint_encoder.n, tensor = share_target)
        tmp = self.fixedpoint_encoder.decode(_pre)
        share = share_target - tmp
        if flag == "inference": self.time_counting_model_inference(share)
        else: self.time_counting(share)
        return tmp, self.cipher.recursive_decrypt(share)

    def secure_Matrix_Multiplication(self, matrix, vector, stage = None, flag = None):
        """
        matrix multiplication with encrypted vector, do vector secret sharing
        """
        if stage == "forward":
            encrypt_vec = self.cipher.recursive_encrypt(vector)
            assert(matrix.shape[1] == encrypt_vec.shape[1])
            mul_result = np.dot(matrix, encrypt_vec.T)
        elif stage == "backward":
            encrypt_vec = self.cipher.recursive_encrypt(vector)
            assert(encrypt_vec.shape[0] == matrix.shape[0])
            mul_result = np.dot(encrypt_vec.T, matrix)
        else: raise NotImplementedError
        return self.secret_share_vector(mul_result, flag)

    def secure_distributed_cal_z(self, X, w1, w2, party = None, flag = None):
        """
        Do the X·w and split into two sharings.
        Args:
            X: ndarray - numpy
            data to use for multiplication
            w1: ndarray ``1 * m1``
            piece 1 of the model weight
            w2: ndarray ``1 * m2``
            piece 2 of the model weight
        Returns:
            Two sharings of the result (X·w)
        """
        if party == "A":
            self._cal_z(X, w1, party = party, encrypt = "paillier")
            assert(X.shape[1] == w2.shape[1])
            if flag == "train": self.time_counting(w2)
            elif flag == "inference": self.time_counting_model_inference(w2)
            self.za2_1, self.za2_2 = self.secure_Matrix_Multiplication(X, w2, stage = "forward", flag = flag)
        elif party == "B":
            self._cal_z(X, w2, party = party, encrypt = "paillier")
            if flag == "train": self.time_counting(w1)
            elif flag == "inference": self.time_counting_model_inference(w1)
            self.zb1_1, self.zb1_2 = self.secure_Matrix_Multiplication(X, w1, stage = "forward", flag = flag)
        else: raise NotImplementedError

    def secure_distributed_compute_loss_cross_entropy(self, label, batch_num):
        """
            Use Taylor series expand log loss:
            Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
            Then loss' = - (1/N)*∑(log(1/2) - 1/2*wx + ywx -1/8(wx)^2)
        """
        half_wx = -0.5 * self.encrypt_wx
        assert(self.encrypt_wx.shape[0] == label.shape[0])
        ywx = self.encrypt_wx * label
        wx_square = (self.za * self.za + 2 * self.za * self.zb + self.zb * self.zb) * -0.125
        loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) - np.log(0.5) )
        return loss

    def Binary_Secure_Classifier_CAESAR(self, X_trainA, X_trainB, X_test1, X_test2, Y_train, Y_test, instances_count, indice_littleside, converge_ondecide):
        self.X_test1 = X_test1
        self.X_test2 = X_test2
        self.Y_test = Y_test

        self.indice = indice_littleside
        if self.data_tag == None: 
            X_batch_listA, X_batch_listB, y_batch_list = self._generate_batch_data_for_distributed_parts(X_trainA, X_trainB, Y_train, self.batch_size)
            self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice])
        elif self.data_tag == 'sparse':
            X_batch_listA, X_batch_listB, y_batch_list = self._generate_Sparse_batch_data_for_distributed_parts(X_trainA, X_trainB, Y_train, self.batch_size)
            self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice])
            # print('Generation done.')
        else: raise Exception("[fit model] No proper entry for batch data generation. Check the data_tag or the fit function.")
        
        """ Train Model """
        self.fit_model_secure_distributed_input(X_batch_listA, X_batch_listB, y_batch_list, instances_count, converge_ondecide)


    def fit_model_secure_distributed_input(self, X_batch_listA, X_batch_listB, y_batch_list, instances_count, converge_ondecide):
        self.n_iteration = 1
        self.loss_history = []
        wa1, wa2 = self.secret_share_vector_plaintext(self.weightA)
        wb1, wb2 = self.secret_share_vector_plaintext(self.weightB)

        flag = "train"
        file = open(self.logname, mode='a+')
        time_start_training_epoch = time.time()
        time_train_counter = 0
        while self.n_iteration <= self.max_iter:
            time_start_training = time.time()
            loss_list = []
            print("[LOGGER] Training epoch:", self.n_iteration)
            for batch_dataA, batch_dataB, batch_labels, batch_num in zip(X_batch_listA, X_batch_listB, y_batch_list, self.batch_num):
                batch_labels = batch_labels.reshape(-1, 1)
                self.secure_distributed_cal_z(X = batch_dataA, w1 = wa1, w2 = wa2, party = "A", flag = flag)
                self.secure_distributed_cal_z(X = batch_dataB, w1 = wb1, w2 = wb2, party = "B", flag = flag)

                self.za = self.za1.T + self.za2_1 + self.zb1_1
                self.time_counting(self.za)
                self.zb = self.zb2.T + self.za2_2 + self.zb1_2

                if self.sigmoid_func == "linear": 
                    encrypt_za = self.cipher.recursive_encrypt(self.za) 
                    self.encrypt_wx = self.zb + encrypt_za
                    self.encrypted_sigmoid_wx = self._compute_sigmoid(self.encrypt_wx)
                elif self.sigmoid_func == "cube":
                    za_first = self.za
                    za_second = self.za * self.za
                    za_third = self.za * self.za * self.za
                    encrypt_za_first = self.cipher.recursive_encrypt(za_first)
                    encrypt_za_second = self.cipher.recursive_encrypt(za_second)
                    encrypt_za_third = self.cipher.recursive_encrypt(za_third)

                    zb_first = self.zb
                    zb_second = self.zb * self.zb
                    zb_third = self.zb * self.zb * self.zb
                    z_first = encrypt_za_first + zb_first
                    self.encrypt_wx = z_first
                    z_third = encrypt_za_third + 3 * encrypt_za_second * zb_first + 3 * encrypt_za_first * zb_second + zb_third
                    self.encrypted_sigmoid_wx = self._compute_sigmoid(z_first, z_third)
                else: 
                    raise NotImplementedError("Invalid sigmoid approximation.")

                # compute error
                self.encrypted_error = (self.encrypted_sigmoid_wx - batch_labels).T
                yb_s, ya_s = self.secret_share_vector(self.encrypted_sigmoid_wx)
                error_b = yb_s - batch_labels
                error_a = ya_s

                # secure backward 
                assert(self.encrypted_error.shape[1] == batch_dataB.shape[0])
                encrypt_gb = np.dot(self.encrypted_error, batch_dataB) * (1 / batch_num)
                gb2, gb1 = self.secret_share_vector(encrypt_gb)

                # Host(A) backward
                ga = np.dot(error_a.T, batch_dataA) * (1 / batch_num)

                error_1_n = error_b * (1 / batch_num)
                self.time_counting(error_1_n)
                ga2_2, ga2_1 = self.secure_Matrix_Multiplication(batch_dataA, error_1_n, stage = "backward")
                assert(self.encrypted_error.shape[1] == batch_dataB.shape[0])

                # compute loss
                batch_loss = self.secure_distributed_compute_loss_cross_entropy(label = batch_labels, batch_num = batch_num)
                loss_list.append(batch_loss)
                
                # update model
                ga_new = ga + ga2_1
                wa1 = wa1 - self.c_penalty * self.alpha * ga_new - self.lambda_para * self.alpha * wa1 / batch_num
                wa2 = wa2 - self.c_penalty * self.alpha * ga2_2 - self.lambda_para * self.alpha * wa2 / batch_num
                wb1 = wb1 - self.c_penalty * self.alpha * gb1 - self.lambda_para * self.alpha * wb1 / batch_num
                wb2 = wb2 - self.c_penalty * self.alpha * gb2 - self.lambda_para * self.alpha * wb2 / batch_num

            # shuffle
            # X_batch_listA, X_batch_listB, y_batch_list = self.shuffle_distributed_data(X_batch_listA, 
            #                     X_batch_listB, y_batch_list)
            
            # sum loss
            loss = np.sum(loss_list) / instances_count
            loss_decrypt = self.cipher.recursive_decrypt(loss)
            time_end_training = time.time()

            if self.ovr == "bin":
                file.write("\nEpoch {}, batch sum loss: {}".format(self.n_iteration, loss_decrypt))
                file.write(" Time: " + str(time_end_training-time_start_training) + "s")
                print("[LOGGER] Epoch {}, batch sum loss: {}".format(self.n_iteration, loss_decrypt), end="")
                print(" Time: " + str(time_end_training-time_start_training) + "s")
            """ 
            intermediate result saving 
            """
            if self.ovr == "bin" and self.n_iteration in self.EPOCH_list:
                time_end_training = time.time()
                epoch_Online_Calculate_timeAccount = time_train_counter
                epoch_Online_Communicate_timeAccount = self.train_commTime_account
                epoch_Online_timeAccounting = epoch_Online_Calculate_timeAccount + epoch_Online_Communicate_timeAccount

                # save Model and Time
                weightA = wa1 + wa2
                weightB = wb1 + wb2
                model_weights = np.hstack((weightA, weightB))
                self.modelWeight_and_Time_List.update({ str(self.n_iteration): [weightA,weightB, epoch_Online_timeAccounting, self.train_commTime_account] })

            elif self.ovr == "ovr" and self.n_iteration in self.EPOCH_list:
                time_end_training = time.time()
                epoch_Online_Calculate_timeAccount = time_train_counter
                epoch_Online_Communicate_timeAccount = self.train_commTime_account
                epoch_Online_timeAccounting = epoch_Online_Calculate_timeAccount + epoch_Online_Communicate_timeAccount
                weightA = wa1 + wa2
                weightB = wb1 + wb2
                model_weights = np.hstack((weightA, weightB))
                self.OVRModel_X.update({ str(self.n_iteration): [weightA,weightB, epoch_Online_timeAccounting, self.train_commTime_account] })

            if (loss_decrypt >= 1e5 or np.isnan(loss_decrypt) or np.isinf(loss_decrypt)):
                file.write("\n[Error] loss overflow.\n")
                self.training_status = "NaN"
                return

            self.is_converged = self.check_converge_by_loss(loss_decrypt)
            if (self.is_converged and converge_ondecide == "on")  or (self.n_iteration == self.max_iter):
                if self.ratio is not None: 
                    self.weightA = wa1 + wa2
                    self.weightB = wb1 + wb2
                    self.model_weights = np.hstack((self.weightA, self.weightB))
                    if self.ovr == "ovr":
                        time_end_training = time.time()
                        file.write("Epoch num: {}, last epoch loss: {}".format(self.n_iteration, loss_decrypt))
                        file.write(" Epoch Total Running Time: {}s, Parallel total run time:{}s".format(str(time_end_training-time_start_training_epoch), str(time_train_counter)))
                        file.write("\nmodel_weights: {}\n".format(self.model_weights))
                break

            self.n_iteration += 1

    def _generate_batch_data_for_distributed_parts(self, X1, X2, y, batch_size):
        """
        Split dataset into each batch.
        """
        print("batch data generating...")
        X_batch_listA = []
        X_batch_listB = []
        y_batch_list = []
        for i in range(len(y) // batch_size):
            X_batch_listA.append(X1[i * batch_size : i * batch_size + batch_size, :])
            X_batch_listB.append(X2[i * batch_size : i * batch_size + batch_size, :])
            y_batch_list.append(y[i * batch_size : i * batch_size + batch_size])
            self.batch_num.append(batch_size)
        if (len(y) % batch_size > 0):
            X_batch_listA.append(X1[len(y) // batch_size * batch_size:, :])
            X_batch_listB.append(X2[len(y) // batch_size * batch_size:, :])
            y_batch_list.append(y[len(y) // batch_size * batch_size:])
            self.batch_num.append(len(y) % batch_size)
        return X_batch_listA, X_batch_listB, y_batch_list

    def _generate_Sparse_batch_data_for_distributed_parts(self, X1, X2, y, batch_size):
        """
        Split sparse dataset into each batch.
        """
        X_batch_listA = []
        X_batch_listB = []
        y_batch_list = []
        X1 = lil_matrix(X1)
        X2 = lil_matrix(X2)
        for i in range(len(y) // batch_size):
            X_batch_listA.append(X1[i * batch_size : i * batch_size + batch_size, :].tocsr())
            X_batch_listB.append(X2[i * batch_size : i * batch_size + batch_size, :].tocsr())
            y_batch_list.append(y[i * batch_size : i * batch_size + batch_size])
            self.batch_num.append(batch_size)

        if (len(y) % batch_size > 0):
            X_batch_listA.append(X1[len(y) // batch_size * batch_size:, :].tocsr())
            X_batch_listB.append(X2[len(y) // batch_size * batch_size:, :].tocsr())
            y_batch_list.append(y[len(y) // batch_size * batch_size:])
            self.batch_num.append(len(y) % batch_size)
        return X_batch_listA, X_batch_listB, y_batch_list

    def predict_distributed_OVR(self, x_test1, x_test2):
        x_test = np.hstack((x_test1, x_test2))
        if self.data_tag == 'sparse':
            z = x_test.dot(self.model_weights.T)
            if not isinstance(z, np.ndarray): z = z.toarray()
        elif self.data_tag == None: z = np.dot(x_test, self.model_weights.T)

        if self.sigmoid_func == "linear": y = self._compute_sigmoid(z)
        elif self.sigmoid_func == "cube": raise NotImplementedError("todo...")
        else: raise NotImplementedError("Invalid sigmoid approximation.")
        return y.reshape(1, -1)

    def OneVsRest_Secure_Classifier_CAESAR(self, X_train1, X_train2, X_test1, X_test2, Y_train, Y_test, converge_ondecide):
        """
        OVR: one vs rest
        """
        self.indice = X_train1.shape[1]
        instances_count = X_train1.shape[0]
        label_lst = list(set(Y_train))
        prob_lst = []

        """ OVR Model Training """
        X_batch_listA, X_batch_listB, y_batch_list = self._generate_batch_data_for_distributed_parts(X_train1, X_train2, 
                                                                                        Y_train, self.batch_size)
        self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice])

        for i in range(len(label_lst)):
            file = open(self.logname, mode='a+')
            pos_label = label_lst[i]
            file.write("\nLabel {} ".format(pos_label))
            def label_reset_OVR(arr):
                return np.where(arr == pos_label, 1, 0)
            y_batch_list_new = []
            y_batch_list_new = list(map(label_reset_OVR, y_batch_list))
            self.weightA[:] = 0
            self.weightB[:] = 0
            self.model_weights[:] = 0

            self.OVRModel_X = dict()
            self.fit_model_secure_distributed_input(X_batch_listA, X_batch_listB, y_batch_list_new, instances_count, converge_ondecide)
            self.OVRModel_Agg.update( {pos_label: self.OVRModel_X} )
            self.train_commTime_account = 0

            if self.EPOCH_list is None:
                prob = self.predict_distributed_OVR(X_test1, X_test2)
                prob = prob.flatten()
                prob_lst.append(prob.tolist())

        self.predict_forEpochs_OVR(X_test1, X_test2, label_lst, Y_test)


    def predict_forEpochs_OVR(self, X_test1, X_test2, label_lst, Y_test):
        """ Epochs for OVR """
        file = open(self.logname, mode='a+')
        accuracy_list = []
        total_Online_commtime_list = []
        total_Online_time_list = []
        total_time_cost_list = []
        inference_time_compute_list = []
        inference_time_total_list = []

        for obj in self.EPOCH_list:
            total_Online_time = 0
            total_Online_commtime = 0
            prob_lst = []
            inference_computer_start = time.time()
            for submodel_iter in range(len(label_lst)):
                pos_label = label_lst[submodel_iter]
                weightA = self.OVRModel_Agg[pos_label][str(obj)][0]
                weightB = self.OVRModel_Agg[pos_label][str(obj)][1]
                total_Online_time += self.OVRModel_Agg[pos_label][str(obj)][2]
                total_Online_commtime += self.OVRModel_Agg[pos_label][str(obj)][3]

                prob = self.predict_base_OVR(X_test1, X_test2, weightA, weightB)
                prob = prob.flatten()
                prob_lst.append(prob.tolist())
            accuracy, score, total_num = self.predict_MAX_OVR(prob_lst, label_lst, Y_test)

            inference_computer_end = time.time()
            self.inference_compute_time = inference_computer_end - inference_computer_start
            accuracy_list.append(accuracy)
            total_Online_commtime_list.append(total_Online_commtime)
            total_Online_time_list.append(total_Online_time)
            total_time_cost_list.append(total_Online_time)
            inference_time_compute_list.append(self.inference_compute_time)
            inference_time_total_list.append(self.inference_time_account + self.inference_compute_time)
            
            # reset
            self.inference_time_account = 0
            self.inference_compute_time = 0


        file.write("\n\nConclusion:")
        file.write("\nEpoch list: {}".format(self.EPOCH_list))
        file.write("\nCAESAR Accuracy List: {}".format(accuracy_list))
        file.write("\nCAESAR total_Online_commtime List: {}".format(total_Online_commtime_list))
        file.write("\nCAESAR total_time_cost List: {}".format(total_time_cost_list))
        file.write("\nCAESAR inference_time_compute List: {}".format(inference_time_compute_list))
        file.write("\nCAESAR inference_time_total List: {}".format(inference_time_total_list))

        file.close()


        """ write down """
        if self.logger is True:
            loggername = ""
            if self.sketch_tag == "sketch":
                loggername = "LOGGER_CAESAR_" + self.dataset_name + "_" + self.kernel_method + "_" + self.sigmoid_func
            elif self.sketch_tag == "raw":
                loggername = "LOGGER_CAESAR_" + self.dataset_name + "_" + "raw" + "_" + self.sigmoid_func
            else: raise NotImplementedError
            
            try:
                File_Path = os.path.join(os.getcwd(), self.dataset_name)
                if not os.path.exists(File_Path): raise FileNotFoundError("Logfile Path not exits.")
                loggername = os.path.join(File_Path, loggername)
            except: raise FileNotFoundError("Logfile Path not exits.")
            
            filelogger = open(loggername + ".txt",  mode="a+")
            
            acc_max = max(accuracy_list, default=None)
            index = accuracy_list.index(acc_max)
            epoch = self.EPOCH_list[index]
            onlineComm = total_Online_commtime_list[index]
            onlinetotal = total_Online_time_list[index]
            totaltime = total_time_cost_list[index]
            infer_compute = inference_time_compute_list[index]
            infer = inference_time_total_list[index]

            filelogger.write("="*71)
            filelogger.write("\nAccuracy: {}, index:{}, epoch:{}, onlineComm:{}s, onlinetotal:{}s, totaltime:{}s, infer_compute:{}s, infer:{}s".format(acc_max, index, epoch, 
                                                                                                                                                    onlineComm, onlinetotal, totaltime, infer_compute, infer))
            filelogger.write("\n    status:{}, origial file name:{}".format(self.training_status, self.logname))
            if self.sketch_tag == "sketch":
                if self.kernel_method in ["pminhash", "0bitcws"]:
                    filelogger.write("\n        batchsize={}, alpha={}, k={}, countsketch_c={}, lambda={}, c_penalty={}, maxiter={}\n\n".format(self.batch_size, self.alpha, 
                                                                                                            self.sampling_k, self.countsketch_c, self.lambda_para, self.c_penalty, 
                                                                                                            self.max_iter))
                else:
                    filelogger.write("\n        batchsize={}, alpha={}, k={}, lambda={}, c_penalty={}, maxiter={}\n\n".format(self.batch_size, self.alpha, 
                                                                                                            self.sampling_k, self.lambda_para, self.c_penalty, 
                                                                                                            self.max_iter))
            elif self.sketch_tag == "raw":
                filelogger.write("\n        batchsize={}, alpha={}, lambda={}, c_penalty={}, scalering_raw={}, maxiter={}\n\n".format(self.batch_size, self.alpha, 
                                                                                                        self.lambda_para, self.c_penalty, self.scalering_raw, 
                                                                                                        self.max_iter))
            filelogger.close()

    def predict_base_OVR(self, X_test1, X_test2, weightA, weightB):
        flag = "inference"

        wa1, wa2 = self.secret_share_vector_plaintext(weightA, flag)
        wb1, wb2 = self.secret_share_vector_plaintext(weightB, flag)

        self.secure_distributed_cal_z(X = X_test1, w1 = wa1, w2 = wa2, party = "A", flag = flag)
        self.secure_distributed_cal_z(X = X_test2, w1 = wb1, w2 = wb2, party = "B", flag = flag)
                
        self.za = self.za1.T + self.za2_1 + self.zb1_1
        self.zb = self.zb2.T + self.za2_2 + self.zb1_2

        # wx
        self._wx = self.za + self.zb
        # sigmoid
        if self.sigmoid_func == "linear": y = self._compute_sigmoid(self._wx)
        elif self.sigmoid_func == "cube":
            za_first = self.za
            za_second = self.za * self.za
            za_third = self.za * self.za * self.za
            encrypt_za_first = self.cipher.recursive_encrypt(za_first)
            encrypt_za_second = self.cipher.recursive_encrypt(za_second)
            encrypt_za_third = self.cipher.recursive_encrypt(za_third)

            zb_first = self.zb
            zb_second = self.zb * self.zb
            zb_third = self.zb * self.zb * self.zb

            z_first = encrypt_za_first + zb_first
            z_third = encrypt_za_third + 3 * encrypt_za_second * zb_first + 3 * encrypt_za_first * zb_second + zb_third
            self.encrypted_sigmoid_wx = self._compute_sigmoid(z_first, z_third)
            y = self.cipher.recursive_decrypt(self.encrypted_sigmoid_wx)
        else: raise NotImplementedError("Invalid sigmoid approximation.")
        return y.reshape(1, -1)


    def predict_MAX_OVR(self, prob_lst, label_lst, Y_test):
        y_predict = []
        prob_array = np.asarray(prob_lst).T

        for i in range(len(Y_test)):
            temp = list(prob_array[i])
            index = temp.index(max(temp))
            y_predict.append(label_lst[index])
        score = 0
        for i in range(len(y_predict)):
            if y_predict[i] == Y_test[i]: score += 1
            else: pass

        total_num = len(y_predict)
        accuracy = float(score)/float(total_num)
        return accuracy, score, total_num


    def predict_base(self, x_test1, x_test2, y_test, weightA, weightB):
        flag = "inference"
        wa1, wa2 = self.secret_share_vector_plaintext(weightA, flag)
        wb1, wb2 = self.secret_share_vector_plaintext(weightB, flag)

        self.secure_distributed_cal_z(X = x_test1, w1 = wa1, w2 = wa2, party = "A", flag = flag)
        self.secure_distributed_cal_z(X = x_test2, w1 = wb1, w2 = wb2, party = "B", flag = flag)
        self.za = self.za1.T + self.za2_1 + self.zb1_1
        self.zb = self.zb2.T + self.za2_2 + self.zb1_2
        # wx
        self._wx = self.za + self.zb
        # sigmoid
        if self.sigmoid_func == "linear":
            y = self._compute_sigmoid(self._wx)

        elif self.sigmoid_func == "cube":
            za_first = self.za
            za_second = self.za * self.za
            za_third = self.za * self.za * self.za
            encrypt_za_first = self.cipher.recursive_encrypt(za_first)
            encrypt_za_second = self.cipher.recursive_encrypt(za_second)
            encrypt_za_third = self.cipher.recursive_encrypt(za_third)
            zb_first = self.zb
            zb_second = self.zb * self.zb
            zb_third = self.zb * self.zb * self.zb
            z_first = encrypt_za_first + zb_first
            z_third = encrypt_za_third + 3 * encrypt_za_second * zb_first + 3 * encrypt_za_first * zb_second + zb_third
            self.encrypted_sigmoid_wx = self._compute_sigmoid(z_first, z_third)
            y = self.cipher.recursive_decrypt(self.encrypted_sigmoid_wx)
        else: 
            raise NotImplementedError("Invalid sigmoid approximation.")
        score = 0
        for i in range(len(y)):
            if y[i] >= 0.5: y[i] = 1
            else: y[i] = 0
            if y[i] == y_test[i]:
                score += 1
            else:
                pass
        total_num = len(y)
        accuracy = float(score)/float(total_num)
        return accuracy, score, total_num


    def predict_forEpochs_Bin(self, x_test1, x_test2, y_test):
        file = open(self.logname, mode='a+')
        weightA = []
        weightB = []
        accuracy_list = []
        total_Online_commtime_list = []
        total_Online_time_list = []
        total_time_cost_list = []
        inference_time_compute_list = []
        inference_time_total_list = []
        for obj in self.EPOCH_list:
            weightA = self.modelWeight_and_Time_List[str(obj)][0]
            weightB = self.modelWeight_and_Time_List[str(obj)][1]
            total_Online_time = self.modelWeight_and_Time_List[str(obj)][2]
            total_Online_commtime = self.modelWeight_and_Time_List[str(obj)][3]

            inference_computer_start = time.time()
            accuracy, score, total_num = self.predict_base(x_test1, x_test2, y_test, weightA, weightB)
            inference_computer_end = time.time()
            self.inference_compute_time = inference_computer_end - inference_computer_start

            accuracy_list.append(accuracy)
            total_Online_commtime_list.append(total_Online_commtime)
            total_Online_time_list.append(total_Online_time)
            total_time_cost_list.append(total_Online_time)
            inference_time_compute_list.append(self.inference_compute_time)
            inference_time_total_list.append(self.inference_time_account + self.inference_compute_time)

            # reset
            self.inference_time_account = 0
            self.inference_compute_time = 0

        file.write("\n\nConclusion:")
        file.write("\nEpoch list: {}".format(self.EPOCH_list))
        file.write("\nCAESAR Accuracy List: {}".format(accuracy_list))
        file.write("\nCAESAR total_Online_commtime List: {}".format(total_Online_commtime_list))
        file.write("\nCAESAR total_time_cost List: {}".format(total_time_cost_list))
        file.write("\nCAESAR inference_time_compute List: {}".format(inference_time_compute_list))
        file.write("\nCAESAR inference_time_total List: {}".format(inference_time_total_list))
        file.close()

        """ write down """
        if self.logger is True:
            loggername = ""
            if self.sketch_tag == "sketch":
                loggername = "LOGGER_CAESAR_" + self.dataset_name + "_" + self.kernel_method + "_" + self.sigmoid_func
            elif self.sketch_tag == "raw":
                loggername = "LOGGER_CAESAR_" + self.dataset_name + "_" + "raw" + "_" + self.sigmoid_func
            else:
                raise NotImplementedError
            
            try:
                File_Path = os.path.join(os.getcwd(), self.dataset_name)
                if not os.path.exists(File_Path):
                    raise FileNotFoundError("Logfile Path not exits.")
                loggername = os.path.join(File_Path, loggername)
            except: raise FileNotFoundError("Logfile Path not exits.")
            filelogger = open(loggername + ".txt",  mode="a+")
            acc_max = max(accuracy_list, default=None)
            index = accuracy_list.index(acc_max)
            epoch = self.EPOCH_list[index]
            onlineComm = total_Online_commtime_list[index]
            onlinetotal = total_Online_time_list[index]
            totaltime = total_time_cost_list[index]
            infer_compute = inference_time_compute_list[index]
            infer = inference_time_total_list[index]
            
            filelogger.write("\nAccuracy: {}, index:{}, epoch:{}, onlineComm:{}s, onlinetotal:{}s, totaltime:{}s, infer_compute:{}s, infer:{}s".format(acc_max, index, epoch, 
                                                                                                                                                    onlineComm, onlinetotal, totaltime, infer_compute, infer))
            filelogger.write("\n        status:{}, origial file name:{}".format(self.training_status, self.logname))
            if self.sketch_tag == "sketch":
                if self.kernel_method in ["pminhash", "0bitcws"]:
                    filelogger.write("\n        batchsize={}, alpha={}, k={}, countsketch_c={}, lambda={}, c_penalty={}, maxiter={}\n\n".format(self.batch_size, self.alpha, 
                                                                                                            self.sampling_k, self.countsketch_c, self.lambda_para, self.c_penalty, 
                                                                                                            self.max_iter))
                else:
                    filelogger.write("\n        batchsize={}, alpha={}, k={}, lambda={}, c_penalty={}, maxiter={}\n\n".format(self.batch_size, self.alpha, 
                                                                                                            self.sampling_k, self.lambda_para, self.c_penalty, 
                                                                                                            self.max_iter))
            elif self.sketch_tag == "raw":
                filelogger.write("\n        batchsize={}, alpha={}, lambda={}, c_penalty={}, scalering_raw={}, maxiter={}\n\n".format(self.batch_size, self.alpha, 
                                                                                                        self.lambda_para, self.c_penalty, self.scalering_raw, 
                                                                                                        self.max_iter))
            filelogger.close()

    def predict_distributed(self, x_test1, x_test2, y_test):
        x_test = np.hstack((x_test1, x_test2))
        if self.data_tag == 'sparse':
            z = x_test.dot(self.model_weights.T)
        elif self.data_tag == None:
            z = np.dot(x_test, self.model_weights.T)
        y = self._compute_sigmoid(z)
        self.score = 0
        for i in range(len(y)):
            if y[i] >= 0.5: y[i] = 1
            else: y[i] = 0
            if y[i] == y_test[i]:
                self.score += 1
            else:
                pass
        self.total_num = len(y)
        self.accuracy = float(self.score)/float(len(y))

def read_distributed_data_raw_or_sketch(dataset_name, raw_or_sketch, kernel_method, portion, sampling_k, ovr, countsketch_, bbitmwhash_, scalering_raw):
    """
    Args:
        dataset_name: 
            name of input dataset
        raw_or_sketch: 
            input data - raw data or sketch data
        kernel_method: 
            pminhash / 0bitcws / rff / poly
        portion: 
            vertically partition scale
        sampling_k: 
            sampling times(k)
        ovr:  
            one vs rest strategy
        scalering_raw: 
            scalering raw data or not
    """
    from sklearn.datasets import load_svmlight_file
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer, RobustScaler
    mm = MinMaxScaler()
    ss = StandardScaler()
    na = Normalizer()
    ma = MaxAbsScaler()
    rs = RobustScaler()

    main_path = PATH_DATA
    if dataset_name in ["ledgar"]:
        dataset_file_name = dataset_name  
        train_file_name = 'ledgar_lexglue_tfidf_train.svm.bz2' 
        test_file_name = 'ledgar_lexglue_tfidf_test.svm.bz2'
    else:
        dataset_file_name = dataset_name  
        train_file_name = dataset_name + '_train.txt' 
        test_file_name = dataset_name + '_test.txt'
    
    rawData_traintype = None # "dense" # "sparse"
    if dataset_name in ["webspam10k_50k", "webspam10k_100k", "webspam10k_500k"]: rawData_traintype = "dense"
    else: rawData_traintype = "dense"

    """
    read Label: Y_train, Y_test
    """
    print("loading dataset...")
    train_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, train_file_name))
    test_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, test_file_name))

    Y_train = train_data[1].astype(int)
    Y_test = test_data[1].astype(int)

    X_train, X_test, Y_train = dataset_selector(dataset_name, train_data, test_data, Y_train)
    
    print("processing dataset...")
    if ovr == "bin":
        if -1 in Y_train:
            Y_train[Y_train != 1] = 0
            Y_test[Y_test != 1] = 0

    if portion == "37": partition = 3/10
    elif portion == "28": partition = 2/10
    elif portion == "19": partition = 1/10
    elif portion == "46": partition = 4/10
    elif portion == "55": partition = 5/10
    else: raise ValueError
    gamma_scale = -1

    if raw_or_sketch == "sketch" and kernel_method in ["pminhash", "0bitcws"]:
        if countsketch_ == 0 and bbitmwhash_ == 0:
            raise ValueError("[error] b(bbitmwhash) and c(countsketch) equals to 0 at the same time!")
        portion_kernel_method = "portion" + portion + "_" + kernel_method
        sketch_sample = "sketch" + sampling_k
        if kernel_method == "pminhash" and countsketch_:
            """ sketch + countsketch """
            dataset_file_name = os.path.join(dataset_name, portion_kernel_method, sketch_sample, "countsketch"+"_"+str(countsketch_))
            train_file_name1 = 'X1_squeeze_train37.txt'
            train_file_name2 = 'X2_squeeze_train37.txt'
            test_file_name1 = 'X1_squeeze_test37.txt'
            test_file_name2 = 'X2_squeeze_test37.txt'
        elif kernel_method == "0bitcws" and countsketch_:
            """ sketch + bbitmwhash """
            dataset_file_name = os.path.join(dataset_name, portion_kernel_method, sketch_sample, "countsketch"+"_"+str(countsketch_))
            train_file_name1 = 'X1_squeeze_train37.txt'
            train_file_name2 = 'X2_squeeze_train37.txt'
            test_file_name1 = 'X1_squeeze_test37.txt'
            test_file_name2 = 'X2_squeeze_test37.txt'
        else:
            raise ValueError("Attempt to read some meaningless data as model training data.")

        X_train1 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name1), delimiter=',')
        X_train2 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name2), delimiter=',')
        X_test1 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name1), delimiter=',')
        X_test2 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name2), delimiter=',')

    elif raw_or_sketch == "sketch" and kernel_method in ["rff", "poly"]:

        X_test = X_test.todense().A

        if dataset_name in ["webspam10k, OVA_Uterus"]:
            portion_kernel_method = "portion" + portion + "_" + kernel_method
            sketch_sample = "sketch" + sampling_k
            dataset_file_name = os.path.join(dataset_name, portion_kernel_method, sketch_sample)
            train_file_name1 = "X1_train.txt"
            train_file_name2 = "X2_train.txt"
            test_file_name1 = "X1_test.txt"
            test_file_name2 = "X2_test.txt"

            if kernel_method == "rff":
                X_train1 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name1), delimiter=',')
                X_train2 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name2), delimiter=',')
                X_test1 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name1), delimiter=',')
                X_test2 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name2), delimiter=',')
                gamma_scale = 1.0000452791983185
            elif kernel_method == "poly":
                X_train1 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name1), delimiter=',')
                X_train2 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name2), delimiter=',')
                X_test1 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name1), delimiter=',')
                X_test2 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name2), delimiter=',')
                gamma_scale = 1.0000452791983185
        else:
            X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale = dataSketch_generator_RBFandPolyKernel(X_train, X_test, Y_train, Y_test, kernel_method, sampling_k, partition)

    elif raw_or_sketch == "raw":
        if rawData_traintype == "dense":
            scalering_list = ["mm", "ss", "na", "ma", "rs"]
            if scalering_raw in scalering_list:
                scaler = eval(scalering_raw)
                X_train = X_train.todense().A
                X_train = scaler.fit_transform(X_train)
                X_test = X_test.todense().A
                X_test = scaler.fit_transform(X_test)
            elif scalering_raw == "nope":
                X_train = X_train.todense().A
                X_test = X_test.todense().A
        elif rawData_traintype == "sparse":
            scalering_list = ["ma"]
            if scalering_raw in scalering_list:
                scaler = eval(scalering_raw)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.fit_transform(X_test)
            elif scalering_raw == "nope": pass
            else: raise TypeError("Sparse data only support MaxAbsScaler, or not using scalering skills.")

        k = X_train.shape[1]
        k1 = np.floor(k * partition).astype(int)
        X_train1, X_train2 = X_train[:,0:k1], X_train[:,k1:]
        k = X_test.shape[1]
        k1 = np.floor(k * partition).astype(int)
        X_test1, X_test2 = X_test[:,0:k1], X_test[:,k1:]
    
    else:
        raise NotImplementedError("Pointing to invalid dataset.")

    return X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale


def logger_info(objectmodel, dataset_name, raw_or_sketch, kernel_method, portion, sampling_k, countsketch_, bbitmwhash_, gamma_scale,
                X_train1_shape, X_train2_shape, X_test1_shape, X_test2_shape, Y_train_shape, Y_test_shape, sigmoid_func, scalering_raw, converge_ondecide):
    file = open(objectmodel.logname, mode='a+')
    file.write("\n =================== # Dataset info # =================== ")
    file.write("\nData source: {} - {}".format(dataset_name, raw_or_sketch))
    file.write("\nFeature: {}".format(objectmodel.ovr)) # bin / ovr
    file.write("\nData Portion: {}".format(portion))

    if raw_or_sketch == "sketch":
        """ sketch data info """
        file.write("\nSketching method: {}".format(kernel_method))
        file.write("\nSampling k: {}".format(sampling_k))

        if kernel_method == "pminhash": file.write("\nUsing Counsketch: c = {}".format(countsketch_))
        elif kernel_method == "0bitcws": file.write("\nUsing Counsketch: c = {}".format(countsketch_))
        elif kernel_method == "rff": file.write("\nUsing rff: gamma = {}".format(gamma_scale))
        elif kernel_method == "poly": file.write("\nUsing poly(TS): gamma = {}".format(gamma_scale))
        else: file.write("\nJust sketch (nope)")
    
    file.write("\nTrain A shape: {}, Train B shape: {}, label shape: {}".format(X_train1_shape, X_train2_shape, Y_train_shape))
    file.write("\nTest data shape: ({}, {}), label shape: {}".format(X_test1_shape[0], X_test1_shape[1]+X_test2_shape[1], Y_test_shape))

    file.write("\n =================== # Training info # =================== ")
    file.write("\nbatch size: {}".format(objectmodel.batch_size))
    file.write("\nalpha: {}".format(objectmodel.alpha))
    file.write("\neps: {}".format(objectmodel.eps))
    file.write("\nlambda: {}".format(objectmodel.lambda_para))
    file.write("\nc_penalty: {}".format(objectmodel.c_penalty))
    file.write("\nmax_iter: {}".format(objectmodel.max_iter))
    file.write("\nWAN_bandwidth: {} Mbps".format(objectmodel.WAN_bandwidth))
    file.write("\nmem_occupancy: {} Byte".format(objectmodel.mem_occupancy))
    file.write("\nsigmoid func: {} approximation".format(sigmoid_func))
    file.write("\nconverge_on_decide: {}".format(converge_ondecide))
    if raw_or_sketch == "raw": file.write("\nscalering_raw: {}".format(scalering_raw))
    abspath = os.path.abspath(__file__)
    file.write("\nPython work_Path : {}".format(abspath))
    file.write("\n =================== #   Info End   # =================== \n\n")

def logger_test_model(objectmodel):
    file = open(objectmodel.logname, mode='a+')
    file.write("\n# ================== #  Test Model  # ================== #")
    file.write("\nscore: {}".format(objectmodel.score))
    file.write("\nlen(y): {}".format(objectmodel.total_num))
    file.write("\nPredict precision: {}\n".format(objectmodel.accuracy))
    file.write("\n# ================== #   Inference Time   # ================== #")
    file.write("\nCAESAR inference_time account: {}s".format(objectmodel.inference_time_account))
    file.close()


def parse_input_parameter():
    """ 
    Desc:
        Initialization parser.

    Tips
        parser.add_argument: `required`.
        should be like `required=True` for almost all parameters here for model init.
        Here set to be False only for the sake of convenience while testing codes.
    """
    parser = argparse.ArgumentParser(description="Parse input parameter to initialize model and dataset.")

    parser.add_argument('-d', '--dataset-name', dest='dataset_name', required=False, type=str, metavar='STRING', help='dataset name')
    parser.add_argument('-p', '--portion', dest='portion', required=False, type=str, choices=['19', '28', '37', '46', '55'], metavar='STRING', help='data division proportion -data-')
    parser.add_argument('-m', '--modeling-method', dest='raw_or_sketch', required=False, type=str, choices=['raw', 'sketch'], metavar='STRING', help='modeling method(decided by input data type) -data-')
    parser.add_argument('-a', '--kernel', dest='kernel_method', required=False, type=str, choices=['pminhash', '0bitcws', 'rff', 'poly'], metavar='STRING', help='kernel approximation method -data-')
    parser.add_argument('-k', '--sampling-k', dest='sampling_k', required=False, type=str, metavar='STRING', help='value `k` for sampling when do kernel approximation.')
    parser.add_argument('-c', '--countsketch', dest='countsketch_', required=False, default = 0, type=int, metavar='INTEGER', help='value c for countsketch method, eg. 2, 4 ...')
    parser.add_argument('-b', '--bbitmwhash', dest='bbitmwhash_', required=False, default = 0, type=int, metavar='INTEGER', help='value b for bbitmwhash method, eg. 1, 2 ...')
    parser.add_argument('-o', '--ovr', dest='ovr', required=False, type=str, choices=['bin', 'ovr'], metavar='STRING', help='training strategy: binary or one vs rest(ovr) classification')
    parser.add_argument('-r', '--scalering-raw', dest='scalering_raw', required=False, type=str, choices=['mm', 'ss', 'na', 'ma', 'rs', 'nope'], metavar='STRING', help='scalering strategy for raw data, `nope` means doing nothing')
    parser.add_argument('-l', '--converge-ondecide', dest='converge_ondecide', required=False, type=str, choices=['on', 'off'], metavar='STRING', help='stopping strategy, `on`-include the loss stopping strategy, `off` otherwise')
    parser.add_argument('-s', '--sigmoid-func', dest='sigmoid_func', required=False, type=str, choices=['linear', 'cube', 'segmentation', 'original'], metavar='STRING', help='sigmoid approximation or original sigmoid')
    parser.add_argument('-al', '--alpha', dest='alpha', required=False, type=float, default = 0.001, metavar='FLOAT', help='learning rate')
    parser.add_argument('-lm', '--lambda-para', dest='lambda_para', required=False, type=float, default = 1.0, metavar='FLOAT', help='lambda parameter for penalty on loss function')
    parser.add_argument('-cp', '--c-penalty', dest='c_penalty', required=False, type=float, default = 1.0, metavar='FLOAT', help='c_penalty parameter for penalty on loss function')
    parser.add_argument('-i', '--max-iter', dest='max_iter', required=False, type=int, metavar='INT', help='max iteraion')
    parser.add_argument('-t', '--batch-size', dest='batch_size', required=False, type=int, default = 20, metavar='INT', help='batch size')
    # parser.add_argument('-e', '--epochlist', dest='EPOCH_list', nargs='*', required=False, type=int, metavar='INT', help='Epoch list for log, zero or more parameters')
    parser.add_argument('-e', '--epochlistmax', dest='Epoch_list_max', required=False, type=int, metavar='INT', help='max epoch num for recordings')
    parser.add_argument('-f', '--logfile-write', dest='Writing_to_Final_Logfile', required=False, default = False, action="store_true", help='write to Final logfile or not, add `-f` is to write, or is not to write')
    args = parser.parse_args()

    return args

    ###
    # -d kits -p 37 -m sketch -a pminhash -k 1024 -c 4 -b 2 -o bin -r mm -l off -s linear -al 0.001 -lm 1 -i 40 -e 40 -f


def LRModelTraining_Mainfunc(dataset_name, portion, raw_or_sketch, kernel_method, sampling_k, 
                                    countsketch_, bbitmwhash_, scalering_raw, ovr, converge_ondecide, alpha, max_iter, 
                                    lambda_para, c_penalty, batch_size, Epoch_list_max, sigmoid_func, Writing_to_Final_Logfile,
                                    X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale):
    """
    Model Training func
    """
    np.random.seed(100)
    weight_vector = np.zeros(X_train1.shape[1]+X_train2.shape[1]).reshape(1, -1)
    LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = batch_size, 
                    max_iter = max_iter, alpha = alpha, eps = 1e-6, ratio = 0.7, penalty = None, lambda_para = lambda_para, 
                    data_tag = None, ovr = ovr, sigmoid_func = sigmoid_func,
                    sketch_tag = raw_or_sketch, countsketch_c = countsketch_, bbitmwhash_b = bbitmwhash_, 
                    dataset_name = dataset_name, kernel_method = kernel_method, sampling_k = sampling_k,
                    Epoch_list_max = Epoch_list_max, logger = Writing_to_Final_Logfile, scalering_raw = scalering_raw, c_penalty = c_penalty)

    logger_info(LogisticRegressionModel, dataset_name, raw_or_sketch, kernel_method, portion, sampling_k, countsketch_, bbitmwhash_, gamma_scale,
                X_train1.shape, X_train2.shape, X_test1.shape, X_test2.shape, Y_train.shape, Y_test.shape, sigmoid_func, scalering_raw, converge_ondecide)

    # model training
    LogisticRegressionModel.time_start_outer = time.time()
    indice_littleside = X_train1.shape[1]
    if LogisticRegressionModel.ovr == "bin":
        LogisticRegressionModel.Binary_Secure_Classifier_CAESAR(X_train1, X_train2, X_test1, X_test2, Y_train, Y_test, X_train1.shape[0], indice_littleside, converge_ondecide)
    elif LogisticRegressionModel.ovr == "ovr":
        LogisticRegressionModel.OneVsRest_Secure_Classifier_CAESAR(X_train1, X_train2, X_test1, X_test2, Y_train, Y_test, converge_ondecide)

    # predict
    if LogisticRegressionModel.ovr == "bin":
        if LogisticRegressionModel.EPOCH_list is not None:
            LogisticRegressionModel.predict_forEpochs_Bin(X_test1, X_test2, Y_test)
        else:
            LogisticRegressionModel.predict_distributed(X_test1, X_test2, Y_test)
            logger_test_model(LogisticRegressionModel)
    elif LogisticRegressionModel.ovr == "ovr": pass
    time_end = time.time()
    file = open(LogisticRegressionModel.logname, mode='a+')
    file.write("\n\n\nTotal time cost of the exhausting running: {}s (≈ {}h)".format(time_end - LogisticRegressionModel.time_start_outer, (time_end - LogisticRegressionModel.time_start_outer) / 3600))
    file.close()


if __name__ == "__main__":
    single_cmdpara_test = True
    """ parameter from scripts """
    args = parse_input_parameter()
    if single_cmdpara_test is True:
        print("[LOGGER] Args come from shell, not default ones.")
        dataset_name = args.dataset_name
        portion = args.portion # 19 / 28 / 37 / 46 / 55
        raw_or_sketch = args.raw_or_sketch # "raw" / "sketch"
        kernel_method = args.kernel_method # 0bitcws / rff / poly
        sampling_k = args.sampling_k
        countsketch_ = args.countsketch_
        bbitmwhash_ = args.bbitmwhash_
        ovr = args.ovr # bin/ovr
        scalering_raw = args.scalering_raw  # mm / ss / na / ma / rs / nope
        converge_ondecide = args.converge_ondecide # on / off
        alpha = args.alpha
        max_iter = args.max_iter
        lambda_para = args.lambda_para
        batch_size = args.batch_size
        c_penalty = args.c_penalty
        Epoch_list_max = args.Epoch_list_max
        sigmoid_func = args.sigmoid_func
        Writing_to_Final_Logfile = args.Writing_to_Final_Logfile
        """ dataset loading """
        X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale = read_distributed_data_raw_or_sketch(dataset_name, raw_or_sketch, 
                                                        kernel_method, portion, sampling_k, ovr, countsketch_, bbitmwhash_, scalering_raw)
        if raw_or_sketch == "sketch" and kernel_method in ["rff", "poly"] and gamma_scale == -1:
            raise ValueError("gamma_scale not updated.")
        LRModelTraining_Mainfunc(dataset_name, portion, raw_or_sketch, kernel_method, sampling_k, 
                                            countsketch_, bbitmwhash_, scalering_raw, ovr, converge_ondecide, alpha, max_iter, 
                                            lambda_para, c_penalty, batch_size, Epoch_list_max, sigmoid_func, Writing_to_Final_Logfile,
                                            X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale)