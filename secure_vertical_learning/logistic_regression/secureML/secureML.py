
import os
import sys
import time
import datetime
import argparse
# from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
import numpy as np
from multiprocessing.pool import Pool

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_pardir = os.path.join(dir_path, os.pardir)
abs_parpardir = os.path.join(abs_pardir, os.pardir)
sys.path.append(abs_pardir)
sys.path.append(abs_parpardir)

from secureModel import SecureLRModel
from paillierm.encrypt import PaillierEncrypt
from paillierm.fixedpoint import FixedPointEndec
from paillierm.utils import urand_tensor
from data_processing.dataset_select import dataset_selector
from data_processing.kernel_approximate_method import dataSketch_generator_RBFandPolyKernel

PATH_DATA = '../../data/'

class SecureML(SecureLRModel):
    """
    SecureML Implementation
    """
    def __init__(self, weight_vector, batch_size, max_iter, alpha, 
                        eps, ratio = None, penalty = None, lambda_para = 1, data_tag = None, ovr = None, sigmoid_func = None,
                        sketch_tag = None, countsketch_c = 0, bbitmwhash_b = 0, dataset_name = None, 
                        kernel_method = None, sampling_k = None, Epoch_list_max = None, logger = None, scalering_raw = None):
        super(SecureML, self).__init__("SecureML")
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
        self.sigmoid_func = sigmoid_func
        self.sampling_k = sampling_k
        self.scalering_raw = scalering_raw

        self.logger = logger
        self.dataset_name = dataset_name
        self.sketch_tag = sketch_tag
        self.training_status = "normal"

        # WAN(Wide area network) Bandwidth
        self.WAN_bandwidth = 10 # Mbps
        self.online_comm_time_account = 0 
        self.offline_comm_time_account = 0
        self.inference_time_account = 0
        self.mem_occupancy = 8 # B
        self.offline_calculate_time = 0

        # enc init
        self.cipher = PaillierEncrypt()
        self.cipher.generate_key()
        self.fixedpoint_encoder = FixedPointEndec(n = 1e10)

        EPOCH_list = []
        EPOCH_list = [i for i in range(1, Epoch_list_max + 1)]
        self.EPOCH_list = EPOCH_list # eg. [1,5,10,15,20,25,30,35,40]
        assert(self.EPOCH_list[-1] <= self.max_iter)
        # Epoch recordings
        if self.ovr == "bin": self.modelWeight_and_Time_List = dict()
        elif self.ovr == "ovr": self.OVRModel_Agg = dict()

        """ time """
        filename = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f')
        if sketch_tag == "sketch":
            if kernel_method == "pminhash":
                self.logname = "SecureML_" + dataset_name + "_" + kernel_method + sampling_k + "_" + str(countsketch_c) + "_" + filename + ".txt"
            elif kernel_method == "0bitcws":
                self.logname = "SecureML_" + dataset_name + "_" + kernel_method + sampling_k + "_" + str(countsketch_c) + "_" + filename + ".txt"
            else:
                self.logname = "SecureML_" + dataset_name + "_" + kernel_method + sampling_k + "_" + filename + ".txt"
        else:
            self.logname = "SecureML_" + dataset_name + "_raw_" + filename + ".txt"
        
        dirName = None
        if sketch_tag == "sketch": dirName = kernel_method
        else: dirName = sketch_tag
        fileAbsPath = os.path.join(os.getcwd(), dataset_name, dirName, self.logname)
        try:
            File_Path = os.path.join(os.getcwd(), dataset_name, dirName)
            if not os.path.exists(File_Path):
                os.makedirs(File_Path)
            self.logname = fileAbsPath
        except: raise FileNotFoundError("Logfile Path not exits.")

    def _sigmoid_segment(self, arr):
        mask = np.logical_and(arr > -0.5, arr < 0.5)
        arr[mask] = arr[mask]
        arr[~mask] = 0
        return arr

    def _compute_sigmoid(self, z):
        return z * 0.25 + 0.5
    
    def _compute_sigmoid_dual_distributed(self, z): 
        return z * 0.25

    def check_converge_by_loss(self, loss):
        converge_flag = False
        if self.pre_loss is None: pass
        elif abs(self.pre_loss - loss) < self.eps:
            converge_flag = True
        self.pre_loss = loss
        return converge_flag
    
    def shuffle_distributed_data(self, XdatalistA, XdatalistB, y_batch_listA, y_batch_listB,
                                 E_batch_list, Z0_batch_list, Z1_batch_list, Z_p0_batch_list, Z_p1_batch_list):
        zip_list = list( zip(XdatalistA, XdatalistB, y_batch_listA, y_batch_listB, 
                             E_batch_list, Z0_batch_list, Z1_batch_list, Z_p0_batch_list, Z_p1_batch_list) )
        np.random.shuffle(zip_list)
        XdatalistA[:], XdatalistB[:], y_batch_listA[:], y_batch_listB[:], E_batch_list[:], Z0_batch_list[:], Z1_batch_list[:], Z_p0_batch_list[:], Z_p1_batch_list[:] = zip(*zip_list)
        return XdatalistA, XdatalistB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, Z_p0_batch_list, Z_p1_batch_list

    def shuffle_lists(self, *lists):
        import random
        """
        Shuffles multiple lists in corresponding order and returns the shuffled lists.

        Args:
            *lists: Variable number of lists to shuffle.

        Returns:
            A tuple containing the shuffled lists in the same order as input.
        """
        n = len(lists[0])
        for lst in lists:
            if len(lst) != n:
                raise ValueError("All lists must have the same length.")
        indices = list(range(n))
        random.shuffle(indices)

        # Shuffle each list using the same indices
        shuffled_lists = []
        for lst in lists:
            shuffled_lists.append([lst[i] for i in indices])
        return tuple(shuffled_lists)

    def time_counting(self, tensor, flag = None):
        commTime = 0
        if flag == "offline":
            if tensor.ndim == 2: object_num = tensor.shape[0] * tensor.shape[1]
            else: object_num = tensor.shape[0]
            commTime = object_num * self.mem_occupancy / (1024*1024) / (self.WAN_bandwidth/8)
            self.offline_comm_time_account += commTime
        else:
            if tensor.ndim == 2: object_num = tensor.shape[0] * tensor.shape[1]
            else: object_num = tensor.shape[0]
            commTime = object_num * self.mem_occupancy / (1024*1024) / (self.WAN_bandwidth/8)
            self.online_comm_time_account += commTime

    def time_counting_model_inference(self, tensor):
        commTime = 0
        if tensor.ndim == 2: object_num = tensor.shape[0] * tensor.shape[1]
        else: object_num = tensor.shape[0]
        commTime = object_num * self.mem_occupancy / (1024*1024) / (self.WAN_bandwidth/8)
        self.inference_time_account += commTime

    def secret_share_vector_plaintext(self, share_target, flag = None):
        _pre = urand_tensor(q_field = self.fixedpoint_encoder.n, tensor = share_target)
        tmp = self.fixedpoint_encoder.decode(_pre)
        share = share_target - tmp
        self.time_counting(share, flag)
        return tmp, share

    def secure_distributed_compute_loss_cross_entropy(self, label, Y_predictA, Y_predictB, batch_num):
        """
        Desc:
            Use Taylor series expand log loss:
            Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
            Then loss' = - (1/N)*∑ ( log(1/2) - 1/2*wx + ywx -1/8(wx)^2 )
        """
        wx = Y_predictA + Y_predictB
        half_wx = -0.5 * wx
        assert(wx.shape[0] == label.shape[0])
        ywx = wx * label
        wx_square = wx * wx * -0.125
        loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) - np.log(0.5) )
        return loss

    def secretSharing_Data_and_Labels(self, data_matrixA, data_matrixB, Y_train):
        local_dataA, share_dataA = self.secret_share_vector_plaintext(data_matrixA)
        local_dataB, share_dataB = self.secret_share_vector_plaintext(data_matrixB)
        local_Y, share_Y = self.secret_share_vector_plaintext(Y_train)

        self.local_matrix_A = np.hstack((local_dataA, share_dataB))
        self.local_matrix_B = np.hstack((share_dataA, local_dataB))
        self.Y_A = local_Y
        self.Y_B = share_Y
        assert(self.local_matrix_A.shape == self.local_matrix_B.shape)
        print("Sharing raw data: \033[32mOK\033[0m")

    def reconstruct(self, Ei, Ei_):
        E = Ei + Ei_
        self.time_counting(Ei)
        self.time_counting(Ei_)
        return E

    def generate_UVZV_Z_multTriplets_beaver_triplets(self, n, d, t, B):
        """
        Generate beaver_triplets and ss to two parties A and B. (Offline phase)
        """
        flag = "offline"
        self.U = np.random.rand(n, d)
        V = np.random.rand(d, t)
        self.U0, self.U1 = self.secret_share_vector_plaintext(self.U, flag)
        self.V0, self.V1 = self.secret_share_vector_plaintext(V, flag)
        self.Z = np.dot(self.U, V)
        self.Z0, self.Z1 = self.secret_share_vector_plaintext(self.Z, flag)

    def _generate_batch_data_and_triples(self, E, batch_size):
        # for two parties in secureML model to generate the batches
        flag = "offline"
        X_batch_listA = []
        X_batch_listB = []
        y_batch_listA = []
        y_batch_listB = []
        E_batch_list = []

        U0_batch_list = []
        U1_batch_list = []

        Z0_batch_list = []
        Z1_batch_list = []
        
        V_p0_batch_list = []
        V_p1_batch_list = []
        Z_p0_batch_list = []
        Z_p1_batch_list = []
        
        for i in range(len(self.Y_A) // batch_size):
            X_batch_listA.append(self.local_matrix_A[i * batch_size : i * batch_size + batch_size, :])
            X_batch_listB.append(self.local_matrix_B[i * batch_size : i * batch_size + batch_size, :])
            y_batch_listA.append(self.Y_A[i * batch_size : i * batch_size + batch_size])
            y_batch_listB.append(self.Y_B[i * batch_size : i * batch_size + batch_size])
            
            E_batch_list.append(E[i * batch_size : i * batch_size + batch_size])
            Z0_batch_list.append(self.Z0[i * batch_size : i * batch_size + batch_size])
            Z1_batch_list.append(self.Z1[i * batch_size : i * batch_size + batch_size])
            U0_batch_list.append(self.U0[i * batch_size : i * batch_size + batch_size])
            U1_batch_list.append(self.U1[i * batch_size : i * batch_size + batch_size])
            self.batch_num.append(batch_size)

            V_p_batch = np.random.rand(batch_size)
            V_p0_, V_p1_ = self.secret_share_vector_plaintext(V_p_batch, flag)
            V_p0_batch_list.append(V_p0_)
            V_p1_batch_list.append(V_p1_)

            Z_p_batch = np.dot(self.U[i * batch_size : i * batch_size + batch_size].T, V_p_batch)
            Z_p0_batch, Z_p1_batch = self.secret_share_vector_plaintext(Z_p_batch, flag)

            Z_p0_batch_list.append(Z_p0_batch)
            Z_p1_batch_list.append(Z_p1_batch)

        if (len(self.Y_A) % batch_size > 0):
            X_batch_listA.append(self.local_matrix_A[len(self.Y_A) // batch_size * batch_size:, :])
            X_batch_listB.append(self.local_matrix_B[len(self.Y_A) // batch_size * batch_size:, :])
            y_batch_listA.append(self.Y_A[len(self.Y_A) // batch_size * batch_size:])
            y_batch_listB.append(self.Y_B[len(self.Y_A) // batch_size * batch_size:])
            
            E_batch_list.append(E[len(self.Y_A) // batch_size * batch_size:])

            Z0_batch_list.append(self.Z0[len(self.Y_A) // batch_size * batch_size:])
            Z1_batch_list.append(self.Z1[len(self.Y_A) // batch_size * batch_size:])
            U0_batch_list.append(self.U0[len(self.Y_A) // batch_size * batch_size:])
            U1_batch_list.append(self.U1[len(self.Y_A) // batch_size * batch_size:])
            self.batch_num.append(len(self.Y_A) % batch_size)

            V_p_batch = np.random.rand(len(self.Y_A) % batch_size)
            V_p0_, V_p1_ = self.secret_share_vector_plaintext(V_p_batch, flag)
            V_p0_batch_list.append(V_p0_)
            V_p1_batch_list.append(V_p1_)
            Z_p_batch = np.dot(self.U[len(self.Y_A) // batch_size * batch_size:].T, V_p_batch)
            Z_p0_batch, Z_p1_batch = self.secret_share_vector_plaintext(Z_p_batch, flag)
            Z_p0_batch_list.append(Z_p0_batch)
            Z_p1_batch_list.append(Z_p1_batch)

        print("Batch data generation: \033[32mOK\033[0m")
        return X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list, V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list

    def fit_model_secure_distributed_input(self, X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, 
                                           E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list, V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list, 
                                           instances_count, converge_ondecide):
        """
        Input: 
            train data(vertically partition)
                        Batch data of Party A, 
                        Batch data of Party B
            label (Secret shared)        
                        Batch y of Party A, 
                        Batch y of Party B
            Masked matrix (E = X - U):                 
                        E_batch_list             
            Triples (Z0 Z1 from Z's share, U0/U1 from U's share)
            instances_count: total instances num
        """

        self.n_iteration = 1
        self.loss_history = []
        self.weightA = self.weightA.reshape(-1, 1)
        self.weightB = self.weightB.reshape(-1, 1)
        file = open(self.logname, mode='a+')
        time_start_training_epoch = time.time()
        while self.n_iteration <= self.max_iter:
            time_start_training = time.time()
            loss_list = []
            batch_label_A = None
            batch_label_B = None
            print("[LOGGER] Training epoch:", self.n_iteration)
            for batch_dataA, batch_dataB, batch_label_A, batch_label_B, batch_E, batch_Z0, batch_Z1, batch_V_p0, batch_V_p1, batch_Z_p0, batch_Z_p1, batch_num in zip(X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, 
                                                                         E_batch_list, Z0_batch_list, Z1_batch_list, V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list, self.batch_num):
                batch_label_A = batch_label_A.reshape(-1, 1)
                batch_label_B = batch_label_B.reshape(-1, 1)

                j = 0
                batch_F0 = self.weightA - self.V0[:,j].reshape(-1, 1)
                batch_F1 = self.weightB - self.V1[:,j].reshape(-1, 1)
                batch_F = self.reconstruct(batch_F0, batch_F1)

                # compute the predict Y*
                Y_predictA = np.dot(batch_dataA, batch_F) + np.dot(batch_E, self.weightA) + batch_Z0[:,j].reshape(-1, 1)
                Y_predictB = np.dot(batch_dataB, batch_F) + np.dot(batch_E, self.weightB) + batch_Z1[:,j].reshape(-1, 1) + -1 * np.dot(batch_E, batch_F)
                
                if self.sigmoid_func == "linear":
                    Y_predictA = self._compute_sigmoid(Y_predictA)
                    Y_predictB = self._compute_sigmoid_dual_distributed(Y_predictB)
                elif self.sigmoid_func == "segmentation":
                    Y = self._sigmoid_segment(Y_predictA + Y_predictB)
                    Y_predictA, Y_predictB = self.secret_share_vector_plaintext(Y)
                else:
                    raise NotImplementedError("Invalid sigmoid approximation.")

                # compute the difference
                batch_D0 = Y_predictA - batch_label_A
                batch_D1 = Y_predictB - batch_label_B

                # backward
                batch_Fp0 = batch_D0 - batch_V_p0.reshape(-1, 1)
                batch_Fp1 = batch_D1 - batch_V_p1.reshape(-1, 1)
                batch_Fp = self.reconstruct(batch_Fp0, batch_Fp1)
                delta0 = np.dot(batch_dataA.T, batch_Fp) + np.dot(batch_E.T, batch_D0) + batch_Z_p0.reshape(-1, 1)
                delta1 = np.dot(batch_dataB.T, batch_Fp) + np.dot(batch_E.T, batch_D1) + batch_Z_p1.reshape(-1, 1) + -1 * np.dot(batch_E.T, batch_Fp)
                
                self.weightA = self.weightA - self.alpha / batch_num * (delta0) -  self.lambda_para * self.alpha * self.weightA / batch_num
                self.weightB = self.weightB - self.alpha / batch_num * (delta1) -  self.lambda_para * self.alpha * self.weightB / batch_num
                j = j + 1

                # compute loss
                batch_loss = self.secure_distributed_compute_loss_cross_entropy(label = batch_label_A + batch_label_B, 
                                                                        Y_predictA=Y_predictA, Y_predictB=Y_predictB, batch_num = batch_num)
                loss_list.append(batch_loss)
            
            # sum loss
            loss = np.sum(loss_list) / instances_count
            print("\rEpoch {}, batch sum loss: {}".format(self.n_iteration, loss), end = '')
            
            time_end_training = time.time()
            if self.ovr == "bin":
                file.write("Epoch {}, batch sum loss: {}".format(self.n_iteration, loss))
                file.write(" Time: " + str(time_end_training-time_start_training) + "s\n")
                print("[LOGGER] Epoch {}, batch sum loss: {}".format(self.n_iteration, loss), end="")
                print(" Time: " + str(time_end_training-time_start_training) + "s")

            """ 
            intermediate result saving 
            """
            if self.ovr == "bin" and self.n_iteration in self.EPOCH_list:
                # Time
                time_end_training = time.time()
                epoch_Online_Calculate_timeAccount = time_end_training - time_start_training_epoch
                epoch_Online_Communicate_timeAccount = self.online_comm_time_account
                epoch_Online_timeAccounting = epoch_Online_Calculate_timeAccount + epoch_Online_Communicate_timeAccount

                ## save Model and Time
                self.modelWeight_and_Time_List.update({ str(self.n_iteration): [self.weightA + self.weightB, epoch_Online_timeAccounting, self.online_comm_time_account] })

            elif self.ovr == "ovr" and self.n_iteration in self.EPOCH_list:
                # Time
                time_end_training = time.time()
                epoch_Online_Calculate_timeAccount = time_end_training - time_start_training_epoch
                epoch_Online_Communicate_timeAccount = self.online_comm_time_account
                epoch_Online_timeAccounting = epoch_Online_Calculate_timeAccount + epoch_Online_Communicate_timeAccount
                self.OVRModel_X.update({ str(self.n_iteration): [self.weightA + self.weightB, epoch_Online_timeAccounting, self.online_comm_time_account] })
                
            if(loss >= 1e5 or np.isnan(loss) or np.isinf(loss)):
                file.write("Epoch num: {}, last epoch loss: {}".format(self.n_iteration, loss))
                file.write("\n[Error] loss overflow.")
                self.training_status = "NaN"
                
            self.is_converged = self.check_converge_by_loss(loss)
            
            if (self.is_converged and converge_ondecide == "on")  or (self.n_iteration == self.max_iter):
                if self.ratio is not None: 
                    self.model_weights = self.weightA + self.weightB
                    time_end_training = time.time()
                    if self.ovr == "ovr":
                        file.write("Epoch num: {}, last epoch loss: {}".format(self.n_iteration, loss))
                        file.write(" Epoch Total Time: " + str(time_end_training-time_start_training_epoch) + "s\n")
                break
            self.n_iteration += 1


    def Binary_Secure_Classifier(self, X_trainA, X_trainB, Y_train, instances_count, feature_count, indice_littleside, converge_ondecide):
        """
        Binary classification
        """
        offline_time_start = time.time()
        print("ratio: ", self.ratio)
        self.indice = indice_littleside

        # generate shared data and labels for two parties
        self.secretSharing_Data_and_Labels(X_trainA, X_trainB, Y_train)

        # split the model weight according to data distribution
        self.weightA = self.model_weights
        self.weightB = self.model_weights

        import math
        t = int(math.ceil(instances_count/self.batch_size))
        print("t: ", t)
        self.generate_UVZV_Z_multTriplets_beaver_triplets(instances_count, feature_count, 
                                                          t, self.batch_size)
        E0 = self.local_matrix_A - self.U0
        E1 = self.local_matrix_B - self.U1
        E = self.reconstruct(E0, E1)

        X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list, V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list = self._generate_batch_data_and_triples(E, self.batch_size)

        offline_time_end = time.time()
        file = open(self.logname, mode='a+')
        file.write("\n =================== # Training Offline Phase # =================== ")
        self.offline_calculate_time += offline_time_end-offline_time_start
        file.write("\nOffline Total Time: " + str(self.offline_calculate_time + self.offline_comm_time_account) + "s")
        file.write("\nOffline Communication Time: " + str(self.offline_comm_time_account) + "s")
        file.write("\n ================= # Training Offline Phase End # ================== \n")

        self.fit_model_secure_distributed_input(X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list, V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list, instances_count, converge_ondecide)

    def predict_distributed_OVR(self, x_test1, x_test2):
        x_test = np.hstack((x_test1, x_test2))
        if self.data_tag == 'sparse':
            z = x_test.dot(self.model_weights.T)
            if not isinstance(z, np.ndarray):
                z = z.toarray()
        elif self.data_tag == None:
            self.model_weights = self.model_weights.reshape(-1, 1)
            z = np.dot(x_test, self.model_weights)

        
        self.time_counting_model_inference(self.model_weights)
        self.time_counting_model_inference(self.model_weights)
        y = self._compute_sigmoid(z)
        return y.reshape(1, -1)


    def y_update_OVR(self, Y_train, batch_size):
        local_Y, share_Y = self.secret_share_vector_plaintext(Y_train)
        self.Y_A = local_Y
        self.Y_B = share_Y

        y_batch_listA = []
        y_batch_listB = []

        for i in range(len(self.Y_A) // batch_size):
            y_batch_listA.append(self.Y_A[i * batch_size : i * batch_size + batch_size])
            y_batch_listB.append(self.Y_B[i * batch_size : i * batch_size + batch_size])
        if (len(self.Y_A) % batch_size > 0):
            y_batch_listA.append(self.Y_A[len(self.Y_A) // batch_size * batch_size:])
            y_batch_listB.append(self.Y_B[len(self.Y_A) // batch_size * batch_size:])
        return y_batch_listA, y_batch_listB


    def OneVsRest_Secure_Classifier(self, X_train1, X_train2, X_test1, X_test2, Y_train, Y_test, converge_ondecide):
        """
        OVR: one vs rest
        """
        offline_time_start = time.time()

        indice_littleside = X_train1.shape[1]
        self.indice = X_train1.shape[1]
        instances_count = X_train1.shape[0]
        label_lst = list(set(Y_train))
        prob_lst = []

        """ OVR Model Training """
        feature_count = X_train1.shape[1]+X_train2.shape[1]
        self.indice = indice_littleside
        self.secretSharing_Data_and_Labels(X_train1, X_train2, Y_train)
        self.weightA = self.model_weights
        self.weightB = self.model_weights
        import math
        t = int(math.ceil(instances_count/self.batch_size))
        self.generate_UVZV_Z_multTriplets_beaver_triplets(instances_count, feature_count, 
                                                          t, self.batch_size)
        E0 = self.local_matrix_A - self.U0
        E1 = self.local_matrix_B - self.U1
        E = self.reconstruct(E0, E1)

        # generate batch data
        X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list, V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list = self._generate_batch_data_and_triples(E, self.batch_size)

        offline_time_end = time.time()
        file = open(self.logname, mode='a+')
        file.write("\n =================== # Training Offline Phase # =================== ")
        self.offline_calculate_time += offline_time_end-offline_time_start
        file.write("\nOffline Total Time: " + str(self.offline_calculate_time + self.offline_comm_time_account) + "s")
        file.write("\nOffline Communication Time: " + str(self.offline_comm_time_account) + "s")
        file.write("\n ================= # Training Offline Phase End # ================== \n")

        x_test = np.hstack((X_test1, X_test2))
        self.time_counting_model_inference(x_test)
        self.time_counting_model_inference(x_test)

        for i in range(len(label_lst)):
            pos_label = label_lst[i]
            file = open(self.logname, mode='a+')
            file.write("Label {}".format(pos_label))

            Y_train_new = np.where(Y_train == pos_label, 1, 0)

            y_batch_listA, y_batch_listB = self.y_update_OVR(Y_train_new, self.batch_size)

            self.weightA = np.zeros(X_train1.shape[1]+X_train2.shape[1]).reshape(-1, 1)
            self.weightB = np.zeros(X_train1.shape[1]+X_train2.shape[1]).reshape(-1, 1)
            self.model_weights = np.zeros(X_train1.shape[1]+X_train2.shape[1]).reshape(-1, 1)

            self.OVRModel_X = dict()
            
            self.fit_model_secure_distributed_input(X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, 
                                                    E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list, 
                                                    V_p0_batch_list, V_p1_batch_list, Z_p0_batch_list, Z_p1_batch_list, 
                                                    instances_count, converge_ondecide)
            
            self.OVRModel_Agg.update( {pos_label: self.OVRModel_X} )
            self.online_comm_time_account = 0
            
            if self.EPOCH_list is None:
                prob = self.predict_distributed_OVR(X_test1, X_test2)
                prob = prob.flatten()
                prob_lst.append(prob.tolist())
        
        if self.EPOCH_list is not None:
            self.predict_forEpochs_OVR(X_test1, X_test2, label_lst, Y_test)

        else:
            print(np.shape(prob_lst))
            y_predict = []
            prob_array = np.asarray(prob_lst).T
            print(prob_array.shape)
            print(type(prob_array))
            print(type(prob_array[0]))
            print(type(prob_array[0][0]))

            for i in range(len(Y_test)):
                temp = list(prob_array[i])
                index = temp.index(max(temp))
                y_predict.append(label_lst[index])
            self.score = 0
            for i in range(len(y_predict)):
                if y_predict[i] == Y_test[i]:
                    self.score += 1
                else:
                    pass
            print("score: ", self.score)
            self.total_num = len(y_predict)
            print("len(y): ", self.total_num)
            self.accuracy = float(self.score)/float(len(y_predict))
            print("\nPredict precision: ", self.accuracy)


    def predict_forEpochs_OVR(self, X_test1, X_test2, label_lst, Y_test):
        file = open(self.logname, mode='a+')
        x_test = np.hstack((X_test1, X_test2))
        # offline
        self.time_counting_model_inference(x_test)
        self.time_counting_model_inference(self.model_weights)
        weight = self.model_weights.reshape(-1, 1)
        z = np.dot(x_test, weight)
        self.time_counting_model_inference(z)
        self.time_counting_model_inference(z)
        self.time_counting_model_inference(z)

        inference_time_offline = self.inference_time_account
        self.inference_time_account = 0

        self.time_counting_model_inference(x_test)
        self.time_counting_model_inference(x_test)
        self.time_counting_model_inference(X_test1)
        self.time_counting_model_inference(X_test2)

        inference_time_base_SSdata = self.inference_time_account
        self.inference_time_account = 0

        accuracy_list = []
        total_Online_commtime_list = []
        total_Online_time_list = []
        total_time_cost_list = []
        inference_time_offline_list = []
        inference_time_compute_list = []
        inference_time_total_list = []

        for obj in self.EPOCH_list:
            total_Online_time = 0
            total_Online_commtime = 0
            prob_lst = []
            inference_computer_start = time.time()

            for submodel_iter in range(len(label_lst)):
                pos_label = label_lst[submodel_iter]
                subModelWeight = self.OVRModel_Agg[pos_label][str(obj)][0]
                total_Online_time += self.OVRModel_Agg[pos_label][str(obj)][1]
                total_Online_commtime += self.OVRModel_Agg[pos_label][str(obj)][2]

                prob = self.predict_base_OVR(X_test1, X_test2, subModelWeight)
                prob = prob.flatten()
                prob_lst.append(prob.tolist())
            accuracy, score, total_num = self.predict_MAX_OVR(prob_lst, label_lst, Y_test)

            inference_computer_end = time.time()
            self.inference_compute_time = inference_computer_end - inference_computer_start

            accuracy_list.append(accuracy)
            total_Online_commtime_list.append(total_Online_commtime)
            total_Online_time_list.append(total_Online_time)
            total_time_cost_list.append(total_Online_time + self.offline_calculate_time + self.offline_comm_time_account)

            inference_time_offline_list.append(inference_time_offline)
            inference_time_compute_list.append(self.inference_compute_time)
            inference_time_total_list.append(self.inference_time_account + inference_time_base_SSdata + inference_time_offline)

            self.inference_time_account = 0
            self.inference_compute_time = 0

        file.write("\n\nConclusion:")
        file.write("\nEpoch list: {}".format(self.EPOCH_list))
        file.write("\nSecureML Accuracy List: {}".format(accuracy_list))
        file.write("\nSecureML total_Online_commtime List: {}".format(total_Online_commtime_list))
        file.write("\nSecureML total_Online_time List: {}".format(total_Online_time_list))
        file.write("\nSecureML total_time_cost List: {}".format(total_time_cost_list))

        file.write("\nSecureML inference_time_offline List: {}".format(inference_time_offline_list))
        file.write("\nSecureML inference_time_compute list: {}".format(inference_time_compute_list))
        file.write("\nSecureML inference_time_total List: {}".format(inference_time_total_list))
        file.close()


        if self.logger is True:
            loggername = ""
            if self.sketch_tag == "sketch":
                loggername = "LOGGER_SecureML_" + self.dataset_name + "_" + self.kernel_method + "_" + self.sigmoid_func
            elif self.sketch_tag == "raw":
                loggername = "LOGGER_SecureML_" + self.dataset_name + "_" + "raw" + "_" + self.sigmoid_func
            else:
                raise NotImplementedError
            try:
                File_Path = os.path.join(os.getcwd(), self.dataset_name)
                if not os.path.exists(File_Path):
                    raise FileNotFoundError("Logfile Path not exits.")
                loggername = os.path.join(File_Path, loggername)
            except:
                raise FileNotFoundError("Logfile Path not exits.")

            filelogger = open(loggername + ".txt",  mode="a+")
            acc_max = max(accuracy_list, default=None)
            index = accuracy_list.index(acc_max)
            epoch = self.EPOCH_list[index]
            onlineComm = total_Online_commtime_list[index]
            onlinetotal = total_Online_time_list[index]
            totaltime = total_time_cost_list[index]
            inferoff = inference_time_offline_list[index]
            infer = inference_time_total_list[index]

            filelogger.write("="*71)
            filelogger.write("\nAccuracy: {}, index:{}, epoch:{}, onlineComm:{}, onlinetotal:{}, totaltime:{}, inferoff:{}, infer:{}".format(acc_max, index, epoch, onlineComm, onlinetotal, totaltime, inferoff, infer))
            filelogger.write("\n        status:{}, origial file name:{}".format(self.training_status, self.logname))
            if self.sketch_tag == "sketch":
                if self.kernel_method in ["pminhash", "0bitcws"]:
                    filelogger.write("\n        batchsize={}, alpha={}, k={}, countsketch_c={}, lambda={}, maxiter={}\n\n".format(self.batch_size, self.alpha, 
                                                                                                            self.sampling_k, self.countsketch_c, self.lambda_para, 
                                                                                                            self.max_iter))
                else:
                    filelogger.write("\n        batchsize={}, alpha={}, k={}, lambda={}, maxiter={}\n\n".format(self.batch_size, self.alpha, 
                                                                                                            self.sampling_k, self.lambda_para, 
                                                                                                            self.max_iter))
            elif self.sketch_tag == "raw":
                filelogger.write("\n        batchsize={}, alpha={}, lambda={}, scalering_raw={}, maxiter={}\n\n".format(self.batch_size, self.alpha, 
                                                                                                        self.lambda_para, self.scalering_raw, 
                                                                                                        self.max_iter))
            filelogger.close()


    def predict_base_OVR(self, X_test1, X_test2, subModelWeight):
        x_test = np.hstack((X_test1, X_test2))
        self.model_weights = subModelWeight.reshape(-1, 1)
        z = np.dot(x_test, self.model_weights)
        self.time_counting_model_inference(self.model_weights)
        self.time_counting_model_inference(self.model_weights)

        if self.sigmoid_func == "linear": y = self._compute_sigmoid(z)
        elif self.sigmoid_func == "segmentation": y = self._sigmoid_segment(z)
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
            if y_predict[i] == Y_test[i]:
                score += 1
            else:
                pass
        print("score: ", score)
        total_num = len(y_predict)
        print("len(y): ", total_num)
        accuracy = float(score)/float(total_num)
        print("\nPredict precision: ", accuracy)
        return accuracy, score, total_num

        

    def predict_distributed(self, x_test1, x_test2, y_test):
        x_test = np.hstack((x_test1, x_test2))
        if self.data_tag == 'sparse':
            z = x_test.dot(self.model_weights.T)
        elif self.data_tag == None:
            self.model_weights = self.model_weights.reshape(-1, 1)
            z = np.dot(x_test, self.model_weights)
        y = self._compute_sigmoid(z)
        self.score = 0
        for i in range(len(y)):
            if y[i] >= 0.5: y[i] = 1
            else: y[i] = 0
            if y[i] == y_test[i]:
                self.score += 1
            else:
                pass
        print("score: ", self.score)
        self.total_num = len(y)
        print("len(y): ", self.total_num)
        self.accuracy = float(self.score)/float(len(y))
        print("\nPredict precision: ", self.accuracy)

    def predict_base(self, x_test1, x_test2, y_test, weight):
        x_test = np.hstack((x_test1, x_test2))
        weight = weight.reshape(-1, 1)
        z = np.dot(x_test, weight)

        self.time_counting_model_inference(weight)
        self.time_counting_model_inference(weight)

        if self.sigmoid_func == "linear":
            y = self._compute_sigmoid(z)
        elif self.sigmoid_func == "segmentation": y = self._sigmoid_segment(z)
        else: raise NotImplementedError("Invalid sigmoid approximation.")
        score = 0
        for i in range(len(y)):
            if y[i] >= 0.5: y[i] = 1
            else: y[i] = 0
            if y[i] == y_test[i]: score += 1
            else: pass
        print("score: ", score)
        total_num = len(y)
        print("len(y): ", total_num)
        accuracy = float(score)/float(total_num)
        print("Predict precision: ", accuracy)
        return accuracy, score, total_num

    def predict_forEpochs_Bin(self, x_test1, x_test2, y_test):
        file = open(self.logname, mode='a+')
        x_test = np.hstack((x_test1, x_test2))
        self.time_counting_model_inference(x_test)
        self.time_counting_model_inference(self.model_weights)
        weight = self.model_weights.reshape(-1, 1)
        z = np.dot(x_test, weight)
        self.time_counting_model_inference(z)
        self.time_counting_model_inference(z)
        self.time_counting_model_inference(z)

        inference_time_offline = self.inference_time_account
        self.inference_time_account = 0

        self.time_counting_model_inference(x_test)
        self.time_counting_model_inference(x_test)
        self.time_counting_model_inference(x_test1)
        self.time_counting_model_inference(x_test2)

        inference_time_base_SSdata = self.inference_time_account
        self.inference_time_account = 0
        accuracy_list = []
        total_Online_commtime_list = []
        total_Online_time_list = []
        total_time_cost_list = []
        inference_time_offline_list = []
        inference_time_compute_list = []
        inference_time_total_list = []

        for obj in self.EPOCH_list:
            weight = self.modelWeight_and_Time_List[str(obj)][0]
            total_Online_time = self.modelWeight_and_Time_List[str(obj)][1]
            total_Online_commtime = self.modelWeight_and_Time_List[str(obj)][2]
            print("Epoch: {}".format(obj))

            inference_computer_start = time.time()
            accuracy, score, total_num = self.predict_base(x_test1, x_test2, y_test, weight)
            inference_computer_end = time.time()
            self.inference_compute_time = inference_computer_end - inference_computer_start

            accuracy_list.append(accuracy)
            total_Online_commtime_list.append(total_Online_commtime)
            total_Online_time_list.append(total_Online_time)
            total_time_cost_list.append(total_Online_time + self.offline_calculate_time + self.offline_comm_time_account)

            inference_time_offline_list.append(inference_time_offline)
            inference_time_compute_list.append(self.inference_compute_time)
            inference_time_total_list.append(self.inference_time_account + inference_time_base_SSdata + inference_time_offline + self.inference_compute_time)

            self.inference_time_account = 0
            self.inference_compute_time = 0

        file.write("\n\nConclusion:")
        file.write("\nEpoch list: {}".format(self.EPOCH_list))
        file.write("\nSecureML Accuracy List: {}".format(accuracy_list))
        file.write("\nSecureML total_Online_commtime List: {}".format(total_Online_commtime_list))
        file.write("\nSecureML total_Online_time List: {}".format(total_Online_time_list))
        file.write("\nSecureML total_time_cost List: {}".format(total_time_cost_list))
        file.write("\nSecureML inference_time_offline List: {}".format(inference_time_offline_list))
        file.write("\nSecureML inference_time_compute list: {}".format(inference_time_compute_list))
        file.write("\nSecureML inference_time_total List: {}".format(inference_time_total_list))
        file.close()

        if self.logger is True:
            loggername = ""
            if self.sketch_tag == "sketch":
                loggername = "LOGGER_SecureML_" + self.dataset_name + "_" + self.kernel_method + "_" + self.sigmoid_func
            elif self.sketch_tag == "raw":
                loggername = "LOGGER_SecureML_" + self.dataset_name + "_" + "raw" + "_" + self.sigmoid_func
            else:
                raise NotImplementedError
            try:
                File_Path = os.path.join(os.getcwd(), self.dataset_name)
                if not os.path.exists(File_Path):
                    raise FileNotFoundError("Logfile Path not exits.")
                loggername = os.path.join(File_Path, loggername)
            except:
                raise FileNotFoundError("Logfile Path not exits.")
            
            filelogger = open(loggername + ".txt",  mode="a+")
            
            acc_max = max(accuracy_list, default=None)
            index = accuracy_list.index(acc_max)
            epoch = self.EPOCH_list[index]
            onlineComm = total_Online_commtime_list[index]
            onlinetotal = total_Online_time_list[index]
            totaltime = total_time_cost_list[index]
            inferoff = inference_time_offline_list[index]
            infer = inference_time_total_list[index]
            
            filelogger.write("="*71)
            filelogger.write("\nAccuracy: {}, index:{}, epoch:{}, onlineComm:{}, onlinetotal:{}, totaltime:{}, inferoff:{}, infer:{}".format(acc_max, index, epoch, onlineComm, onlinetotal, totaltime, inferoff, infer))
            filelogger.write("\n        status:{}, origial file name:{}".format(self.training_status, self.logname))
            if self.sketch_tag == "sketch":
                if self.kernel_method in ["pminhash", "0bitcws"]:
                    filelogger.write("\n        batchsize={}, alpha={}, k={}, countsketch_c={}, lambda={}, maxiter={}\n\n".format(self.batch_size, self.alpha, 
                                                                                                            self.sampling_k, self.countsketch_c, self.lambda_para, 
                                                                                                            self.max_iter))
                else:
                    filelogger.write("\n        batchsize={}, alpha={}, k={}, lambda={}, maxiter={}\n\n".format(self.batch_size, self.alpha, 
                                                                                                            self.sampling_k, self.lambda_para, 
                                                                                                            self.max_iter))
            elif self.sketch_tag == "raw":
                filelogger.write("\n        batchsize={}, alpha={}, lambda={}, scalering_raw={}, maxiter={}\n\n".format(self.batch_size, self.alpha, 
                                                                                                        self.lambda_para, self.scalering_raw, 
                                                                                                        self.max_iter))
            filelogger.close()



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
    dataset_file_name = dataset_name  
    train_file_name = dataset_name + '_train.txt' 
    test_file_name = dataset_name + '_test.txt'
    
    rawData_traintype = None # "dense" # "sparse"
    if dataset_name in ["webspam10k_50k", "webspam10k_100k", "webspam10k_500k"]:
        rawData_traintype = "dense"
    else:
        rawData_traintype = "dense"

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
        X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale = dataSketch_generator_RBFandPolyKernel(X_train, X_test, Y_train, Y_test, kernel_method, sampling_k, partition)

    elif raw_or_sketch == "raw":
        print("Try to read Raw data...")
        if rawData_traintype == "dense":
            scalering_list = ["mm", "ss", "na", "ma", "rs"]
            if scalering_raw in scalering_list:
                scaler = eval(scalering_raw)
                X_train = X_train.todense().A
                X_train = scaler.fit_transform(X_train)
                X_test = X_test.todense().A
                X_test = scaler.fit_transform(X_test)
            elif scalering_raw == "nope":
                print("To dense, [nope]")
                X_train = X_train.todense().A
                X_test = X_test.todense().A
        elif rawData_traintype == "sparse":
            scalering_list = ["ma"]
            if scalering_raw in scalering_list:
                scaler = eval(scalering_raw)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.fit_transform(X_test)
            elif scalering_raw == "nope":
                print("To dense, [nope]")
                pass
            else:
                raise TypeError("Sparse data only support MaxAbsScaler, or not using scalering skills.")
        k = X_train.shape[1]
        k1 = np.floor(k * partition).astype(int)
        X_train1, X_train2 = X_train[:,0:k1], X_train[:,k1:]
        k = X_test.shape[1]
        k1 = np.floor(k * partition).astype(int)
        X_test1, X_test2 = X_test[:,0:k1], X_test[:,k1:]
    else: raise NotImplementedError("Pointing to invalid dataset.")
    
    return X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale



def logger_info(objectmodel, dataset_name, raw_or_sketch, kernel_method, portion, sampling_k, countsketch_, bbitmwhash_, gamma_scale,
                X_train1_shape, X_train2_shape, X_test1_shape, X_test2_shape, Y_train_shape, Y_test_shape, sigmoid_func, scalering_raw, converge_ondecide):
    file = open(objectmodel.logname, mode='a+')
    file.write("\n =================== # Dataset info # =================== ")
    file.write("\nData source: {} - {}".format(dataset_name, raw_or_sketch))
    file.write("\nFeature: {}".format(objectmodel.ovr))
    file.write("\nData Portion: {}".format(portion))
    if raw_or_sketch == "sketch":
        """ sketch data info """
        file.write("\nSketching method: {}".format(kernel_method))
        file.write("\nSampling k: {}".format(sampling_k))
        if kernel_method == "pminhash": 
            file.write("\nUsing Counsketch: c = {}".format(countsketch_))
        elif kernel_method == "0bitcws": 
            file.write("\nUsing Counsketch: c = {}".format(countsketch_))
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
    file.write("\n\nPredict precision: {}\n".format(objectmodel.accuracy))

    file.write("\n# ================== #   Inference Time   # ================== #")
    file.write("\nSecureMLModel inference_time account: {}s".format(objectmodel.inference_time_account))
    file.close()


def parse_input_parameter():
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
    parser.add_argument('-i', '--max-iter', dest='max_iter', required=False, type=int, metavar='INT', help='max iteraion')
    parser.add_argument('-t', '--batch-size', dest='batch_size', required=False, type=int, default = 20, metavar='INT', help='batch size')
    # parser.add_argument('-e', '--epochlist', dest='EPOCH_list', nargs='*', required=False, type=int, metavar='INT', help='Epoch list for log, zero or more parameters')
    parser.add_argument('-e', '--epochlistmax', dest='Epoch_list_max', required=False, type=int, metavar='INT', help='max epoch num for recordings')
    parser.add_argument('-f', '--logfile-write', dest='Writing_to_Final_Logfile', required=False, default = False, action="store_true", help='write to Final logfile or not, add `-f` is to write, or is not to write')
    args = parser.parse_args()

    return args
    ###
    # -d kits -p 37 -m sketch -a pminhash -k 1024 -c 4 -b 2 -o bin -r mm -l off -s linear -al 0.01 -lm 1 -i 40 -t 20 -e 1 5 10 15 20 25 30 35 40 -f

def LRModelTraining_Mainfunc(dataset_name, portion, raw_or_sketch, kernel_method, sampling_k, 
                                    countsketch_, bbitmwhash_, scalering_raw, ovr, converge_ondecide, alpha, max_iter, 
                                    lambda_para, batch_size, Epoch_list_max, sigmoid_func, Writing_to_Final_Logfile,
                                    X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale):

    weight_vector = np.zeros(X_train1.shape[1]+X_train2.shape[1]).reshape(1, -1)
    SecureMLModel = SecureML(weight_vector = weight_vector, batch_size = batch_size, 
                    max_iter = max_iter, alpha = alpha, eps = 1e-5, ratio = 0.7, penalty = None, lambda_para = lambda_para, 
                    data_tag = None, ovr = ovr, sigmoid_func = sigmoid_func,
                    sketch_tag = raw_or_sketch, countsketch_c = countsketch_, bbitmwhash_b = bbitmwhash_, 
                    dataset_name = dataset_name, kernel_method = kernel_method, sampling_k = sampling_k,
                    Epoch_list_max = Epoch_list_max, logger = Writing_to_Final_Logfile, scalering_raw = scalering_raw)

    logger_info(SecureMLModel, dataset_name, raw_or_sketch, kernel_method, portion, sampling_k, countsketch_, bbitmwhash_, gamma_scale,
                X_train1.shape, X_train2.shape, X_test1.shape, X_test2.shape, Y_train.shape, Y_test.shape, sigmoid_func, scalering_raw, converge_ondecide)
    time_start = time.time()
    indice_littleside = X_train1.shape[1]
    if SecureMLModel.ovr == "bin":
        SecureMLModel.Binary_Secure_Classifier(X_train1, X_train2, Y_train, X_train1.shape[0], (X_train1.shape[1]+X_train2.shape[1]), indice_littleside, converge_ondecide)
    elif SecureMLModel.ovr == "ovr":
        SecureMLModel.OneVsRest_Secure_Classifier(X_train1, X_train2, X_test1, X_test2, Y_train, Y_test, converge_ondecide)
    if SecureMLModel.ovr == "bin":
        if SecureMLModel.EPOCH_list is not None:
            SecureMLModel.predict_forEpochs_Bin(X_test1, X_test2, Y_test)
        else:
            SecureMLModel.predict_distributed(X_test1, X_test2, Y_test)
            logger_test_model(SecureMLModel)
    elif SecureMLModel.ovr == "ovr": pass
    time_end = time.time()
    file = open(SecureMLModel.logname, mode='a+')
    file.write("\n\n\nTotal time cost of the exhausting running: {}s (≈ {}h)".format(time_end - time_start, (time_end - time_start) / 3600))
    file.close()


if __name__ == "__main__":
    local_test = False
    single_cmdpara_test = True
    args = parse_input_parameter()
    if single_cmdpara_test is True:
        print("[LOGGER] args come from shell, not default ones.")

        dataset_name = args.dataset_name
        portion = args.portion
        raw_or_sketch = args.raw_or_sketch # "raw" / "sketch"
        kernel_method = args.kernel_method # 0bitcws / rff / poly
        sampling_k = args.sampling_k
        countsketch_ = args.countsketch_
        bbitmwhash_ = args.bbitmwhash_
        ovr = args.ovr
        scalering_raw = args.scalering_raw
        converge_ondecide = args.converge_ondecide
        alpha = args.alpha
        max_iter = args.max_iter
        lambda_para = args.lambda_para
        batch_size = args.batch_size
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
                                            lambda_para, batch_size, Epoch_list_max, sigmoid_func, Writing_to_Final_Logfile,
                                            X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale)
    elif local_test == True:
        print("[LOGGER] args are default ones.")
        dataset_name = "kits"
        portion = "37"
        raw_or_sketch = "sketch"
        kernel_method = "0bitcws"
        sampling_k = "1024"
        countsketch_ = 2
        bbitmwhash_ = 2

        ovr = "bin"
        scalering_raw = "mm"
        converge_ondecide = "off"
        #common parameters
        alpha = 0.001
        max_iter = 5
        Epoch_list_max = 5
        lambda_para = 1
        batch_size = 16
        sigmoid_func = "linear"
        Writing_to_Final_Logfile = True
        """ dataset loading """
        X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale = read_distributed_data_raw_or_sketch(dataset_name, raw_or_sketch, 
                                                        kernel_method, portion, sampling_k, ovr, countsketch_, bbitmwhash_, scalering_raw)
        if raw_or_sketch == "sketch" and kernel_method in ["rff", "poly"] and gamma_scale == -1:
            raise ValueError("gamma_scale not updated.")
        LRModelTraining_Mainfunc(dataset_name, portion, raw_or_sketch, kernel_method, sampling_k, 
                                    countsketch_, bbitmwhash_, scalering_raw, ovr, converge_ondecide, alpha, max_iter, 
                                    lambda_para, batch_size, Epoch_list_max, sigmoid_func, Writing_to_Final_Logfile,
                                    X_train1, X_train2, Y_train, X_test1, X_test2, Y_test, gamma_scale)