import numpy as np
import scipy.sparse as sp
import os
from sklearn.datasets import load_svmlight_file

# distributed sample to sketch
def Dis_samples_to_sketch2(m, n1, n2, k1, k2, b, c, samples1, samples2):
    """
    Args:
        m: Number of samples
        n: Number of features
        n1: Maximum number of features in the first part of the dataset after feature division
        k: Number of samples for P-minhash and 0-bit CWS (the sum of columns of samples1 and samples2)
        b: b in b-bit minwise hash (compresses the number of features n to 2^b)
        c: c in Count Sketch (compresses the number of features n to c)
        samples1: CWS sampling results of the first part of the dataset
        samples2: CWS sampling results of the second part of the dataset

    Returns: 
        Convert the samples obtained from P-minhash and 0-bit CWS to sketches
    """

    assert m == samples1.shape[0]
    assert m == samples2.shape[0]
    assert k1 == samples1.shape[1]
    assert k2 == samples2.shape[1] 
    # 1 w/o compression
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
    # 2 Count Sketch
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

    # 3 b-bit minwise hash
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
    else: return 'ERROR'



PATH_DATA = '../data/'

def sample_to_countsketch(tag, b, c, dataset, kernel_approx, portion, sampling_rate, sketching_method):
    """
    dataset path template:
        dataset_name/portion<portion>_<method>/<sampling_>/(4 dataset file)
    
    read samples, turn into sketch (including countsketch/0bitminwisehash compression or w/o compression)
    """
    if tag == "train": print("================= Train =================")
    else: print("================= Test =================")
    print("sketch-{}, k={}; method-{}, b={}, c={}".format(kernel_approx, sampling_rate, sketching_method, b, c))

    sketch_name = "sketch" + str(sampling_rate) # sketch1024 or sketch512
    portion_method = "portion37" + "_" + str(kernel_approx) # portion37_pminhash / portion37_0bitcws
    dataset_file_name = os.path.join(dataset, portion_method, sketch_name)
    # example: dataset_file_name = "DailySports/portion37_pminhash/sketch1024/""
    
    train_file_name1 = 'X1_train_samples.txt'
    train_file_name2 = 'X2_train_samples.txt'
    test_file_name1 = 'X1_test_samples.txt'
    test_file_name2 = 'X2_test_samples.txt'
    main_path = PATH_DATA

    if portion == "37": partition = 3/10
    elif portion == "28": partition = 2/10
    elif portion == "19": partition = 1/10
    elif portion == "46": partition = 4/10
    elif portion == "55": partition = 5/10
    else: raise ValueError

    # kernel_approx = "pminhash"  # pminhash / 0bitcws / rff / poly
    # if kernel_approx == "pminhash": # countsketch
    #     assert(b == 0)
    #     assert(sketching_method == "countsketch")
    # elif kernel_approx == "0bitcws": # b-bit minwise hash
    #     assert(c == 0)
    #     assert(sketching_method == "bbitmwhash")

    """ load raw dataset """
    main_path = PATH_DATA
    # dataset_Rawfile_name = dataset  
    # train_Rawfile_name = dataset + '_train.txt' 
    # test_Rawfile_name = dataset + '_test.txt'
    if dataset_name in ["ledgar"]:
        dataset_Rawfile_name = dataset  
        train_Rawfile_name = 'ledgar_lexglue_tfidf_train.svm.bz2' 
        test_Rawfile_name = 'ledgar_lexglue_tfidf_test.svm.bz2'
    else:
        dataset_file_name = dataset_name  
        train_Rawfile_name = dataset_name + '_train.txt' 
        test_Rawfile_name = dataset_name + '_test.txt'

    """ train dataset """
    if tag == "train":
        X_train_samples_1 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name1), delimiter=',', dtype = int)
        X_train_samples_2 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name2), delimiter=',', dtype = int)
        train_data = load_svmlight_file(os.path.join(main_path, dataset_Rawfile_name, train_Rawfile_name))

        m = X_train_samples_1.shape[0]
        k = X_train_samples_1.shape[1] + X_train_samples_2.shape[1]
        k1 = X_train_samples_1.shape[1]
        k2 = X_train_samples_2.shape[1]

        n = train_data[0].shape[1]
        n1 = np.floor(n * partition).astype(int)
        n2 = n - n1
        print("Origin train data (samples) shape: ", m, n)
        print("k = {}, c = {}, b = {}".format(k, c, b))
        
        """ Method 2 for countsketch (Pminhash, 0bitcws) """
        sketch = Dis_samples_to_sketch2(m, n1, n2, k1, k2, b, c, X_train_samples_1, X_train_samples_2).toarray()
        print("Sketch train data shape: ", sketch.shape)

        sk = sketch.shape[1]
        # partition = 3/10
        sk1 = np.floor(sk * partition).astype(int)
        result_X_train1, result_X_train2 = sketch[:,0:sk1], sketch[:,sk1:]

        if sketching_method == "countsketch":
            assert(sketch.shape[0] == m)
            assert(sketch.shape[1] == (c * k))
            train_sketch_savepath = str(sketching_method) + "_" + str(c) # "countsketch"
            train_save_name1 = "X1_squeeze_train37.txt"
            train_save_name2 = "X2_squeeze_train37.txt"
            """
            # PATH eg.:  ../data/DailySports/portion37_pminhash/sketch1024/countsketch_2/
            """
            np.savetxt(os.path.join(main_path, dataset_file_name, train_sketch_savepath, train_save_name1),
                        result_X_train1, delimiter=',', fmt='%d')
            np.savetxt(os.path.join(main_path, dataset_file_name, train_sketch_savepath, train_save_name2),
                        result_X_train2, delimiter=',', fmt='%d')

        elif sketching_method == "bbitmwhash":
            assert(sketch.shape[0] == m)
            assert(sketch.shape[1] == ((2 ** b) * k))
            train_sketch_savepath = str(sketching_method) + "_" + str(2 ** b) # "bbitmwhash" b = [1,2,3]; 2**b=[2^1 2^2 2^3]=[2 4 8]
            train_save_name1 = "X1_squeeze_train37.txt"
            train_save_name2 = "X2_squeeze_train37.txt"

            """
            # PATH eg.:  ../data/ DailySports/portion37_0bitcws/sketch1024/ bbitmwhash_2/
            """
            np.savetxt(os.path.join(main_path, dataset_file_name, train_sketch_savepath, train_save_name1),
                        result_X_train1, delimiter=',', fmt='%d')
            np.savetxt(os.path.join(main_path, dataset_file_name, train_sketch_savepath, train_save_name2),
                        result_X_train2, delimiter=',', fmt='%d')

        print("Train count-sketch data saved.")
        
    elif tag == "test":
        X_test_samples_1 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name1), delimiter=',', dtype = int)
        X_test_samples_2 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name2), delimiter=',', dtype = int)
        test_data = load_svmlight_file(os.path.join(main_path, dataset_Rawfile_name, test_Rawfile_name))

        m = X_test_samples_1.shape[0]
        k = X_test_samples_1.shape[1] + X_test_samples_2.shape[1]
        k1 = X_test_samples_1.shape[1]
        k2 = X_test_samples_2.shape[1]

        n = test_data[0].shape[1]
        n1 = np.floor(n * partition).astype(int)
        n2 = n - n1
        print("Origin test data (samples) shape: ", m, n)
        print("k = {}, c = {}, b = {}".format(k, c, b))

        sketch = Dis_samples_to_sketch2(m, n1, n2, k1, k2, b, c, X_test_samples_1, X_test_samples_2).toarray()
        print("Sketch test data shape: ", sketch.shape)

        """ split the dataset according to ratio """
        sk = sketch.shape[1]
        sk1 = np.floor(sk * partition).astype(int)
        result_X_test1, result_X_test2 = sketch[:,0:sk1], sketch[:,sk1:]

        print(result_X_test1.shape[0], result_X_test1.shape[1])
        print(result_X_test2.shape[0], result_X_test2.shape[1])

        if sketching_method == "countsketch":
            assert(sketch.shape[0] == m)
            assert(sketch.shape[1] == (c * k))
            test_sketch_savepath = str(sketching_method) + "_" + str(c)  # "countsketch"
            test_save_name1 = "X1_squeeze_test37.txt"
            test_save_name2 = "X2_squeeze_test37.txt"
            """
            # PATH eg.:  ../data/DailySports/portion37_pminhash/sketch1024/countsketch_2/
            """
            np.savetxt(os.path.join(main_path, dataset_file_name, test_sketch_savepath, test_save_name1),
                        result_X_test1, delimiter=',', fmt='%d')
            np.savetxt(os.path.join(main_path, dataset_file_name, test_sketch_savepath, test_save_name2),
                        result_X_test2, delimiter=',', fmt='%d')
            

        elif sketching_method == "bbitmwhash":
            assert(sketch.shape[0] == m)
            assert(sketch.shape[1] == ((2 ** b) * k))
            test_sketch_savepath = str(sketching_method) + "_" + str(2 ** b)  # "bbitmwhash" b = [1,2,3]; 2**b=[2^1 2^2 2^3]=[2 4 8]
            test_save_name1 = "X1_squeeze_test37.txt"
            test_save_name2 = "X2_squeeze_test37.txt"
            """
            # PATH eg.:  ../data/DailySports/portion37_0bitcws/sketch1024/bbitmwhash_2/
            """
            np.savetxt(os.path.join(main_path, dataset_file_name, test_sketch_savepath, test_save_name1),
                        result_X_test1, delimiter=',', fmt='%d')
            np.savetxt(os.path.join(main_path, dataset_file_name, test_sketch_savepath, test_save_name2),
                        result_X_test2, delimiter=',', fmt='%d')

        print("Test count-sketch data saved.")
        print("========================================")
    else:
        raise Exception('[Exception] tag error occurred.')
    

if __name__ == '__main__':
    # X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = read_distributed_encoded_data()
    print("loading dataset...")
    dataset_name = "kits" # DailySports (ok) / kits / robert ok / cifar10 ok / SVHN (ok) / webspam10k / cifar10_total / ledgar
    print("dataset: ", dataset_name)
    portion = "37"
    for kernel_approx in ["pminhash", "0bitcws"]:
        for sampling_rate in [1024]:
            if kernel_approx == "pminhash": 
                sketching_method = "countsketch"
                b = 0
                for c in [8]: 
                # for c in [2, 4]:
                    sample_to_countsketch("train", b, c, dataset_name, kernel_approx, portion, sampling_rate, sketching_method)
                    sample_to_countsketch("test", b, c, dataset_name, kernel_approx, portion, sampling_rate, sketching_method)

            elif kernel_approx == "0bitcws": 
                # sketching_method = "bbitmwhash"
                sketching_method = "countsketch"
                # c = 0
                # # for b in [3, 4]: 
                # for b in [1, 2]:
                b = 0
                for c in [8]: 
                # for c in [2, 4]:
                # for c in [2, 4]:
                    sample_to_countsketch("train", b, c, dataset_name, kernel_approx, portion, sampling_rate, sketching_method)
                    sample_to_countsketch("test", b, c, dataset_name, kernel_approx, portion, sampling_rate, sketching_method)