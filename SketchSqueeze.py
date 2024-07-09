import numpy as np
import scipy.sparse as sp


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
        return 'ERROR'



def Dis_samples_to_sketch(m, n, n1, k, b, c, samples1, samples2):


    assert m == samples1.shape[0]                        
    assert m == samples2.shape[0]
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
        return 'ERROR'