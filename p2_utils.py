import pandas as pd
import numpy as np
import matplotlib
import warnings
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats as st
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.neural_network import MLPRegressor
import torch

from scipy.stats import norm
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

class Standardize:
    def __init__(self):
        self.mean = 0
        self.std = 0
        
    def fit(self, data):
        self.mean = data.mean()
        self.std = data.std(ddof=1)
        return (data-self.mean)/self.std
    
    def transform(self, data):
        return (data-self.mean)/self.std
    
    def inverse(self, data):
        return (data*self.std)+self.mean
    
class InnerCVDataCollection:
    def __init__(self, n_param):
        self.perf = np.zeros((1,n_param))
        
    def set_perf(self, error, idx):
        self.perf[0,idx] = error
        
    def get_perf(self):
        return self.perf
    
class OuterCVDataCollection:
    def __init__(self, K):
        self.bm_perf = np.zeros((1,K))
        self.ann_perf = np.zeros((1,K))
        self.ann_param_idx = np.zeros((1,K))
        self.lm_perf = np.zeros((1,K))
        self.lm_param_idx = np.zeros((1,K))
        
    def set_bm_perf(self, error, idx):
        self.bm_perf[0,idx] = error

    def set_ann_perf(self, error, idx):
        self.ann_perf[0,idx] = error
        
    def set_ann_param_idx(self, param_idx, idx):
        self.ann_param_idx[0,idx] = param_idx
        
    def set_lm_perf(self, error, idx):
        self.lm_perf[0,idx] = error
        
    def set_lm_param_idx(self, param_idx, idx):
        self.lm_param_idx[0,idx] = param_idx
        
    def get_bm_perf(self):
        return self.bm_perf
    
    def get_ann_perf(self):
        return self.ann_perf
    
    def get_ann_param_idx(self):
        return self.ann_param_idx
    
    def get_lm_perf(self):
        return self.lm_perf
    
    def get_lm_param_idx(self):
        return self.lm_param_idx
        
def get_best_parameters(inner_CV_list, param, param_idx):
    avg_sum = 0
    for q in range(len(inner_CV_list)):
        avg_sum = avg_sum + inner_CV_list[q][param_idx].get_perf()
    gen_errors = avg_sum/len(inner_CV_list)
        
    best_param_idx = np.argmin(gen_errors)
    best_param = param[best_param_idx]

    return best_param_idx, best_param

def l1_loss(y_pred, y_true):
    return np.abs(y_true-y_pred)

def l2_loss(y_pred, y_true):
    return np.square(y_true-y_pred)

def mse(y_pred, y_true):
    return l2_loss(y_pred, y_true).sum()/y_pred.shape[0]