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

from p2_classification import classification
from p2_regression import regression_a, regression_b
from p2_utils import Standardize, InnerCVDataCollection, OuterCVDataCollection, get_best_parameters, mse, l1_loss, l2_loss

from scipy.stats import norm
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)    
            
def main():
    ### load data ###
    data_path = os.path.join(os.getcwd(), "LAozone.csv")
    data = pd.read_csv(data_path)

    ### add additional feature ###
    # decoding seasons from doy
    # 0 = winter (december, january, february)
    # 1 = spring (march, april, may)
    # 2 = summer (june, july, august)
    # 3 = autumn (september, october, november)
    data["season"] = 0
    for row in data.index:
        if data["doy"][row] <= 60 or data["doy"][row] > 335:
            data["season"][row] = 0
        if data["doy"][row] > 60 and data["doy"][row] <= 152:
            data["season"][row] = 1
        if data["doy"][row] > 152 and data["doy"][row] <= 244:
            data["season"][row] = 2
        if data["doy"][row] > 244 and data["doy"][row] <= 335:
            data["season"][row] = 3
            
    data_regr = data.drop(["doy", "season"], axis=1)
    data_class = data.drop(["doy"], axis=1)
    
    X_regr = data_regr.drop(["temp"], axis=1)
    y_regr = data_regr["temp"]
    
    X_class = data_class.drop(["season"], axis=1)
    y_class = data_class["season"]
    
    regression_a(X_regr, y_regr)
    
    regression_b(X_regr, y_regr)

    classification(X_class, y_class)
        
    
    
    
if __name__ == "__main__":
    main()
    
    