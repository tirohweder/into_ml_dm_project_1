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

def regression_a(X_regr, y_regr):
    ### Regression part A ###
    
    #initialize CV
    K = 10
    CV = KFold(n_splits=K, shuffle=True, random_state=44)
    
    #list of parameters for regression model
    reg_param = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    
    #array to record all errors
    error_mat = np.empty((K,len(reg_param)))
    
    #CV loop
    j = 0
    for train_idx, test_idx in CV.split(X_regr):
        #initialization of training ad test data for the split
        X_train, y_train = X_regr.values[train_idx,:], y_regr.values[train_idx]
        X_test, y_test = X_regr.values[test_idx,:], y_regr.values[test_idx]
        
        #shall this be done for every CV or just once at the begining?
        #standardization of the training data
        standardizer = Standardize()
        X_train = standardizer.fit(X_train)
        X_test = standardizer.transform(X_test)
        
        i = 0
        for param in reg_param:
            #model needs to be adapted once seen in the lecture
            #training of linear regression model
            lm = Lasso(alpha=param)
            lm.fit(X_train,y_train)
            
            #prediction of linear regression model
            y_pred_train = lm.predict(X_train)
            y_pred_test = lm.predict(X_test)
            
            #error calculation
            error_train = np.square(y_train-y_pred_train).sum()/y_train.shape[0]
            error_test = np.square(y_test-y_pred_test).sum()/y_test.shape[0]
            
            error_mat[j,i] = error_test
            
            i = i+1
        
        j = j+1
    
    #calculation of generalization error
    gen_error_mat = np.empty((1,len(reg_param)))
    for l in range(error_mat.shape[1]):
        gen_error_mat[0,l] = error_mat[:,l].mean()
    
    #plot the generalization error w.r.t the regularization parameter
    plt.figure()
    plt.plot(np.asarray(reg_param).reshape(1,len(reg_param)),gen_error_mat, "bo")
    plt.title('Generalization Error as a function of the regularization parameter');
    plt.xlabel('regularization parameter');
    plt.ylabel('Generalization error');
    plt.xscale("log")
    plt.show()
    
    #determine best model (according to generalization error)
    best_model = np.argmin(gen_error_mat, axis=1)
    
    #redo CV for the best model to generate avg. of coefficients
    coef_list = []
    for train_idx, test_idx in CV.split(X_regr):
        #initialization of training ad test data for the split
        X_train, y_train = X_regr.values[train_idx,:], y_regr.values[train_idx]
        X_test, y_test = X_regr.values[test_idx,:], y_regr.values[test_idx]
        
        #shall this be done for every CV or just once at the begining?
        #standardization of the training data
        standardizer = Standardize()
        X_train = standardizer.fit(X_train)
        X_test = standardizer.transform(X_test)
        
        #model needs to be adapted once seen in the lecture
        #training of linear regression model
        lm = Lasso(alpha=np.ndarray.item(np.asarray(reg_param, dtype=float)[best_model]))
        lm.fit(X_train,y_train)
        coef_list.append(lm.coef_)  
    
    #calculate mean of coefficients
    coef_array = np.asarray(coef_list)
    coef_mean = np.zeros((1,coef_array.shape[1]))
    for i in range(coef_array.shape[1]):
        coef_mean[0,i] = coef_array[:,i].mean()

    #bar plot for the mean coefficients w.r.t the features
    plt.figure()
    plt.barh(np.asarray(X_regr.columns), coef_mean.flatten())
    plt.title('Mean Coefficients of Best Linear Regression Model');
    plt.xlabel('mean coefficient');
    plt.ylabel('feature');
    plt.show()
    
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
        
    
    
    
if __name__ == "__main__":
    main()
    
    