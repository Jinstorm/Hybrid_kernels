"""
Parameters
----------
data: {array-like, sparse matrix}, shape (n_instances, n_features)
    a data matrix where row represents feature and column is data instance
dimension_num: int
    the length of hash code
seed: int, default: 1
    part of the seed of the random number generator
Returns
-----------
fingerprints_k: ndarray, shape (n_instances, dimension_num)
    one component of hash code from some algorithms, and each row is the hash code for a data instance
fingerprints_y: ndarray, shape (n_instances, dimension_num)
    one component of hash code from some algorithms, and each row is the hash code for a data instance
fingerprints: ndarray, shape (n_instances, dimension_num)
    only one component of hash code from some algorithms, and each row is the hash code for a data instance
elapsed: float
    time of hashing data matrix
"""


import numpy as np
import numpy.matlib
import scipy as sp
import scipy.sparse as sparse
import time
import gc


class WeightedMinHash:
    """
    Attributes:
    -----------
    PRIME_RANGE: int
        the range of prime numbers

    weighted_set: {array-like, sparse matrix}, shape (n_instances, n_features)
        a data matrix where row represents feature and column is data instance
    dimension_num: int
        the length of hash code(in other word, the times of CWS, i.e. k)
    seed: int, default: 1
        part of the seed of the random number generator. Note that the random seed consists of seed and repeat.
    instance_num: int
        the number of data instances
    feature_num: int
        the number of features
    """


    def __init__(self, weighted_set, dimension_num, seed=1):

        self.weighted_set = weighted_set
        self.dimension_num = dimension_num
        self.seed = seed
        self.instance_num = self.weighted_set.shape[0]                
        self.feature_num = self.weighted_set.shape[1]               


    def licws(self, repeat=1):
        """The 0-bit Consistent Weighted Sampling (0-bit CWS) algorithm generates the original hash code $(k, y_k)$
           by running ICWS, but finally adopts only $k$ to constitute the fingerprint.
           P. Li, "0-bit Consistent Weighted Sampling", in KDD, 2015, pp. 665-674.
        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator
        Returns
        ----------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance
        elapsed: float
            time of hashing data matrix
        """

        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = np.random.uniform(0, 1, (self.dimension_num, self.feature_num))

        v1 = np.random.uniform(0, 1, (self.dimension_num, self.feature_num))
   
        v2 = np.random.uniform(0, 1, (self.dimension_num, self.feature_num))

        u1 = np.random.uniform(0, 1, (self.dimension_num, self.feature_num))
    
        u2 = np.random.uniform(0, 1, (self.dimension_num, self.feature_num))

        zero_num = 0 
        for i_sample in range(0, self.instance_num):
            t1 = time.time()
            feature_id = sparse.find(self.weighted_set[i_sample, :] > 0)[1]
            if list(feature_id) == []:
                fingerprints[i_sample, :] = np.full(self.dimension_num, -1)
                zero_num += 1
            else:
                gamma = - np.log(np.multiply(u1[:, feature_id], u2[:, feature_id]))
                t_matrix = np.floor(np.divide(
                    np.matlib.repmat(np.log(self.weighted_set[i_sample, feature_id].todense()), self.dimension_num, 1),
                    gamma) + beta[:, feature_id])
                y_matrix = np.exp(np.multiply(gamma, t_matrix - beta[:, feature_id]))
                a_matrix = np.divide(np.multiply(-np.log(np.multiply(v1[:, feature_id], v2[:, feature_id])),
                                                 np.multiply(u1[:, feature_id], u2[:, feature_id])), y_matrix)
                min_position = np.argmin(a_matrix, axis=1).T                       
                fingerprints[i_sample, :] = feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed, zero_num



    def P_minhash(self, repeat=1):
        """
         Parameters
         ----------
         repeat: int, default: 1
             the number of repeating the algorithm as the part of the seed of the random number generator
         Returns
         ----------
         fingerprints: ndarray, shape (n_instances, dimension_num)
             hash codes for data matrix, where row represents a data instance
         elapsed: float
             time of hashing data matrix
         """

        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        u = np.random.uniform(0, 1, (self.dimension_num, self.feature_num))
        zero_num = 0
        for i_sample in range(0, self.instance_num):
            feature_id = sparse.find(self.weighted_set[i_sample, :] > 0)[1]
            if list(feature_id) == []:
                fingerprints[i_sample, :] = np.full(self.dimension_num, -1)
                zero_num += 1
            else:
                c_matrix = np.divide(np.log(u[:, feature_id]), np.matlib.repmat(self.weighted_set[i_sample, feature_id].todense(), self.dimension_num, 1)) * (-1.0)
                min_position = np.argmin(c_matrix, axis=1).T
                fingerprints[i_sample, :] = feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed, zero_num
