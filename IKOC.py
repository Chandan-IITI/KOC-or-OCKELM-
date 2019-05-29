
# Author: Chandan Gautam
# Institute: IIT Indore, India
# Email: chandangautam31@gmail.com , phd1501101001@iiti.ac.in


import numpy as np
from cvxopt import matrix
from cvxopt import sparse
from cvxopt.solvers import qp
import math


def linear_kernel(X, Y=None):
    return np.dot(X, X.T) if Y is None else np.dot(X, Y.T)


class IKOC(object):
    '''
    This class implements KOC+ algorithm,
    '''
    def __init__(self, nu, features_kernel=linear_kernel,
                 privileged_kernel=linear_kernel,
                 privileged_regularization=0.1,
                 regularization=0, tol=0.001):

        self.nu = nu
        self.tol = tol
        self.features_kernel = features_kernel
        self.privileged_kernel = privileged_kernel
        self.regularization = regularization
        self.privileged_regularization = privileged_regularization
        self.dual_solution = None
        self.support_indices = None
        self.support_vectors = None
        self.dual_alpha = None


    def fit(self, X, Z):
        '''
        Method takes matrix with feature values (X)
        and information from privilaged feature
        space (Z), compute output weight beta
        '''
        kernel_x = self.features_kernel(X)
        kernel_z = self.privileged_kernel(Z)
        if self.regularization==0:
            C = 1.0 / len(X) / self.nu         # Like SVDD or ISVDD
        else:
            C = self.regularization           # Fix the nu at 0.05 or 0.10 and vary C
        Cpriv = self.privileged_regularization
        size = X.shape[0]
        T = np.ones((size, 1))
        self.dual_alpha = np.dot(np.dot(np.linalg.inv(Cpriv * kernel_x + C * (np.dot(kernel_x, kernel_z)) + kernel_z),(np.eye(size)*Cpriv + C * kernel_z)), T)
        self.support_vectors = X
        score1 = abs(T - np.dot(kernel_x,self.dual_alpha))
        #score1_sort = score1.sort()
        score1_sort = sorted(score1) 
        fracrej = self.nu
        self.threshold = score1_sort[int(math.ceil(size*(1-fracrej)))-1]
        self.score_train = self.threshold - score1
        return self

    def decision_function(self, X):
        """
        Return anomaly score for points in X
        """
        T_one = np.ones((X.shape[0], 1))
        score_test = self.threshold - abs(T_one - np.dot(self.features_kernel(X, self.support_vectors),self.dual_alpha)) 
        
        return score_test
    
class KOC(object):
    '''
    This class implements KOC+ algorithm,
    '''
    def __init__(self, nu, features_kernel=linear_kernel,
                 regularization=0, tol=0.001):

        self.nu = nu
        self.tol = tol
        self.features_kernel = features_kernel
        self.regularization = regularization
        self.dual_solution = None
        self.support_indices = None
        self.support_vectors = None
        self.dual_alpha = None


    def fit(self, X):
        '''
        Method takes matrix with feature values (X)
        and information from privilaged feature
        space (Z), compute output weight beta
        '''
        kernel_x = self.features_kernel(X)
        if self.regularization==0:
            C = 1.0 / len(X) / self.nu         # Like SVDD or ISVDD
        else:
            C = self.regularization           # Fix the nu at 0.05 or 0.10 and vary C
        size = X.shape[0]
        T = np.ones((size, 1))
        self.dual_alpha = np.dot(np.linalg.inv(np.eye(size)*C + kernel_x), T)
        self.support_vectors = X
        score1 = abs(T - np.dot(kernel_x,self.dual_alpha))
        #score1_sort = score1.sort()
        score1_sort = sorted(score1) 
        fracrej = self.nu
        self.threshold = score1_sort[int(math.ceil(size*(1-fracrej)))-1]
        self.score_train = self.threshold - score1
        return self
    
    def decision_function(self, X):
        """
        Return anomaly score for points in X
        """
        T_one = np.ones((X.shape[0], 1))
        score_test = self.threshold - abs(T_one - np.dot(self.features_kernel(X, self.support_vectors),self.dual_alpha)) 
        
        return score_test
