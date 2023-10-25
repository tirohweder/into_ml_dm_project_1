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
    
class BaselineRegression:
    def __init__(self):
        self.prediction = 0
        
    def fit(self, X, y):
        self.prediction = y.mean()
        
    def predict(self, X):
        return np.ravel(self.prediction*np.ones((X.shape[0],1)))

class L2RegularizedLinearRegression:
    def __init__(self, lambda_, bias):
        self.bias = bias
        self.lambda_ = lambda_

    def fit(self, X, y):
        if self.bias == True:
            X = np.concatenate((np.ones((X.shape[0],1)),X),1)
            lambdaI = self.lambda_ * np.eye(X.shape[1])
            lambdaI[0,0] = 0
        elif self.bias == False:
            lambdaI = self.lambda_ * np.eye(X.shape[1])

        self.w = np.linalg.solve(X.T @ X + lambdaI, X.T @ y)


    def predict(self, X):
        if self.bias == True:
            X = np.concatenate((np.ones((X.shape[0],1)),X),1)

        return X @ self.w

    def get_weights(self):
        return self.w

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
        self.mlp_perf = np.zeros((1,K))
        self.mlp_param_idx = np.zeros((1,K))
        self.lm_perf = np.zeros((1,K))
        self.lm_param_idx = np.zeros((1,K))
        
    def set_bm_perf(self, error, idx):
        self.bm_perf[0,idx] = error
        
    def set_mlp_perf(self, error, idx):
        self.mlp_perf[0,idx] = error
        
    def set_mlp_param_idx(self, param_idx, idx):
        self.mlp_param_idx[0,idx] = param_idx
        
    def set_lm_perf(self, error, idx):
        self.lm_perf[0,idx] = error
        
    def set_lm_param_idx(self, param_idx, idx):
        self.lm_param_idx[0,idx] = param_idx
        
    def get_bm_perf(self):
        return self.bm_perf
    
    def get_mlp_perf(self):
        return self.mlp_perf
    
    def get_mlp_param_idx(self):
        return self.mlp_param_idx
    
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

def paired_t_test(z):
    CI = st.t.interval(1 - 0.05, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p = 2*st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
    return CI, p

def regression_a(X_regr, y_regr):
    ### Regression part A ###
    
    #initialize CV
    K = 10
    CV = KFold(n_splits=K, shuffle=True, random_state=44)
    
    #list of parameters for regression model
    reg_param = [0, 10**-15, 10**-12, 10**-9, 10**-6, 10**-3, 10**-2, 0.1, 1]
    bias = True
    
    #array to record all errors
    error_mat = np.empty((K,len(reg_param)))
    error_train_mat = np.empty((K,len(reg_param)))

    #array to record all weights
    weight_mat = []
    
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
        
        weight_list = []

        i = 0
        for param in reg_param:
            #model needs to be adapted once seen in the lecture
            #training of linear regression model
            lm = L2RegularizedLinearRegression(bias=bias, lambda_ = param)
            lm.fit(X_train,y_train)
            
            #prediction of linear regression model
            y_pred_train = lm.predict(X_train)
            y_pred_test = lm.predict(X_test)
            
            #error calculation
            error_train = np.square(y_train-y_pred_train).sum()/y_train.shape[0]
            error_test = np.square(y_test-y_pred_test).sum()/y_test.shape[0]
            
            error_train_mat[j,i] = error_train
            error_mat[j,i] = error_test

            weight_list.append(lm.get_weights())
            
            i = i+1
        
        weight_mat.append(weight_list)

        j = j+1

    print(weight_mat[1][2])
    #calculation of generalization error
    gen_error_mat = np.empty((1,len(reg_param)))
    for l in range(error_mat.shape[1]):
        gen_error_mat[0,l] = error_mat[:,l].mean()

    #calculation of average training error
    avg_train_error_mat = np.empty((1,len(reg_param)))
    for l in range(error_train_mat.shape[1]):
        avg_train_error_mat[0,l] = error_train_mat[:,l].mean()
    
    #determine best model (according to generalization error)
    best_model = np.argmin(gen_error_mat, axis=1)
    print(f"The best model parameter lambda is: {np.ndarray.item(np.asarray(reg_param, dtype=float)[best_model])}")
    print(f"The overall best model is: {np.where(error_mat == error_mat.min())} ")

    #plot the generalization error w.r.t the regularization parameter
    plt.figure()
    plt.plot(np.asarray(reg_param).ravel(), avg_train_error_mat.ravel(), "r.-", label="Avg. Training Error")
    plt.plot(np.asarray(reg_param).ravel(), gen_error_mat.ravel(), "b.-", label="Generalization Error")
    plt.title('Generalization Error as a function of the regularization parameter')
    plt.xlabel('regularization parameter')
    plt.ylabel('Generalization error')
    plt.xscale("log")
    plt.legend()
    plt.show()

    #calculate mean of coefficients for best model
    coef_mean_list = []
    for j in range(len(weight_mat[0])):
        weight_sum = np.zeros((1,weight_mat[0][0].shape[0]))
        for i in range(len(weight_mat)):
            weight_sum = weight_mat[i][j] + weight_sum

        coef_mean_list.append(weight_sum/len(weight_mat[0]))

    coef_mean = coef_mean_list[best_model[0]]

    #bar plot for the mean coefficients w.r.t the features
    plt.figure()
    if bias == False:
        plt.barh(np.asarray(X_regr.columns), coef_mean.flatten())
    else:
        plt.barh(np.concatenate((["bias"],np.asarray(X_regr.columns)),0), coef_mean.flatten())
    plt.title('Mean Coefficients of Best Linear Regression Model')
    plt.xlabel('mean coefficient')
    plt.ylabel('feature')
    plt.show()
    
def regression_b(X_regr, y_regr):
    ### Regression part A ###
    
    #initialize CV
    outer_K = 10
    outer_CV = KFold(n_splits=outer_K, shuffle=True, random_state=44)
    
    inner_K = 10
    inner_CV = KFold(n_splits=inner_K, shuffle=True, random_state=44)
    
    #create list of arrays to store all inner CV data
    inner_CV_list = []
    
    #create outer collector to store outer CV data
    outer_collector = OuterCVDataCollection(outer_K)
    
    #define MLP parameters
    mlp_param = [(1,), (5,5,), (30,20,10,), (20,40,20,), (10,20,20,10,)]
    
    #define lineare regression parameters
    lm_param = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    
    #initialize loss lists
    z_l1_bm = []
    z_l2_bm = []
    z_l1_mlp = []
    z_l2_mlp = []
    z_l1_lm = []
    z_l2_lm = []

    #outer CV loop
    j=0
    for par_idx, test_idx in outer_CV.split(X_regr):
        #initialization of training ad test data for the split
        X_par, y_par = X_regr.values[par_idx,:], y_regr.values[par_idx]
        X_test, y_test = X_regr.values[test_idx,:], y_regr.values[test_idx]
        
        #inner CV loop
        i=0
        for train_idx, val_idx in inner_CV.split(X_par):
            #initialization of training ad test data for the split
            X_train, y_train = X_par[train_idx,:], y_par[train_idx]
            X_val, y_val = X_par[val_idx,:], y_par[val_idx]
            
            #shall this be done for every CV or just once at the begining?
            #standardization of the training data
            standardizer = Standardize()
            X_train = standardizer.fit(X_train)
            X_val = standardizer.transform(X_val)
            
            #inner_collector = InnerCVDataCollection(len(mlp_param), len(lm_param))
            inner_collector_bm = InnerCVDataCollection(1)
            inner_collector_mlp = InnerCVDataCollection(len(mlp_param))
            inner_collector_lm = InnerCVDataCollection(len(lm_param))
            
            #fit baseline model
            bm = BaselineRegression()
            bm.fit(X_train,y_train)
            
            y_train_pred_bm = bm.predict(X_train)
            y_val_pred_bm = bm.predict(X_val)
            
            error_train_bm = np.square(y_train-y_train_pred_bm).sum()/y_train.shape[0]
            error_val_bm = np.square(y_val-y_val_pred_bm).sum()/y_val.shape[0]
            
            inner_collector_bm.set_perf(error_val_bm,0)
            
            p=0
            for param in mlp_param:
                #model needs to be adapted once seen in the lecture
                #training of ann model
                mlp = MLPRegressor(hidden_layer_sizes=param, max_iter=2500, learning_rate_init=0.01)
                mlp.fit(X_train,y_train)
                
                #prediction of linear regression model
                y_train_pred_mlp = mlp.predict(X_train)
                y_val_pred_mlp = mlp.predict(X_val)
                
                error_train_mlp = np.square(y_train-y_train_pred_mlp).sum()/y_train.shape[0]
                error_val_mlp = np.square(y_val-y_val_pred_mlp).sum()/y_val.shape[0]
                
                inner_collector_mlp.set_perf(error_val_mlp, p)
                
                p=p+1
                
            p=0    
            for param in lm_param:
                #model needs to be adapted once seen in the lecture
                #training of linear regression model
                lm = Lasso(alpha=param)
                lm.fit(X_train,y_train)
                
                #prediction of linear regression model
                y_train_pred_lm = lm.predict(X_train)
                y_val_pred_lm = lm.predict(X_val)
                
                #error calculation
                error_train_lm = np.square(y_train-y_train_pred_lm).sum()/y_train.shape[0]
                error_val_lm = np.square(y_val-y_val_pred_lm).sum()/y_val.shape[0]
                
                inner_collector_lm.set_perf(error_val_lm, p)
                
                p=p+1
             
            inner_CV_list.append([inner_collector_bm, inner_collector_mlp, inner_collector_lm])
            
            i=i+1
        
        #standardize the data
        standardizer = Standardize()
        X_par = standardizer.fit(X_par)
        X_test = standardizer.transform(X_test)
        
        #train best basemodel of inner CV
        bm = BaselineRegression()
        bm.fit(X_par,y_par)
        
        y_par_pred_bm = bm.predict(X_par)
        y_test_pred_bm = bm.predict(X_test)

        z_l1_bm.append(np.transpose(l1_loss(y_test_pred_bm, y_test)))
        z_l2_bm.append(np.transpose(l2_loss(y_test_pred_bm, y_test)))
        
        mse_par_bm = mse(y_par_pred_bm, y_par)
        mse_test_bm = mse(y_test_pred_bm, y_test)
        
        outer_collector.set_bm_perf(mse_test_bm, j)
        
        #train best MLP of inner CV
        best_param_idx, best_param = get_best_parameters(inner_CV_list, mlp_param, 1)
        
        mlp = MLPRegressor(hidden_layer_sizes=best_param, max_iter=2500, learning_rate_init=0.01)
        mlp.fit(X_par,y_par)
        
        y_par_pred_mlp = mlp.predict(X_par)
        y_test_pred_mlp = mlp.predict(X_test)

        z_l1_mlp.append(np.transpose(l1_loss(y_test_pred_mlp, y_test)))
        z_l2_mlp.append(np.transpose(l2_loss(y_test_pred_mlp, y_test)))
        
        mse_par_mlp = mse(y_par_pred_mlp, y_par)
        mse_test_mlp = mse(y_test_pred_mlp, y_test)
        
        outer_collector.set_mlp_perf(mse_test_mlp, j)
        outer_collector.set_mlp_param_idx(best_param_idx, j)
        
        #train best Linear Model of inner CV
        best_param_idx, best_param = get_best_parameters(inner_CV_list, lm_param, 2)
        
        lm = Lasso(alpha=best_param)
        lm.fit(X_par,y_par)
        
        y_par_pred_lm = lm.predict(X_par)
        y_test_pred_lm = lm.predict(X_test)

        z_l1_lm.append(np.transpose(l1_loss(y_test_pred_lm, y_test)))
        z_l2_lm.append(np.transpose(l2_loss(y_test_pred_lm, y_test)))
        
        mse_par_lm = mse(y_par_pred_lm, y_par)
        mse_test_lm = mse(y_test_pred_lm, y_test)
        
        outer_collector.set_lm_perf(mse_test_lm, j)
        outer_collector.set_lm_param_idx(best_param_idx, j)

        j=j+1
    
    #create summarization table for the outer CV loop
    table_b_df = pd.DataFrame(columns=["ANN_param", "ANN_error", "LM_param", "LM_error", "BM_error"], index=range(outer_K))       
    for r in range(outer_K):
        table_b_df["ANN_param"][r] = mlp_param[int(outer_collector.get_mlp_param_idx()[0,r])]
        table_b_df["ANN_error"][r] = outer_collector.get_mlp_perf()[0,r]
        table_b_df["LM_param"][r] = lm_param[int(outer_collector.get_lm_param_idx()[0,r])]
        table_b_df["LM_error"][r] = outer_collector.get_lm_perf()[0,r]
        table_b_df["BM_error"][r] = outer_collector.get_bm_perf()[0,r]
    
    with open(os.path.join(os.getcwd(),'regression_b_table.txt'), 'w') as f:
        f.write(table_b_df.to_string())

    #setup I statistical analysis based on l2 loss

    z_l2_bm = np.concatenate(z_l2_bm)
    z_l2_mlp = np.concatenate(z_l2_mlp)
    z_l2_lm = np.concatenate(z_l2_lm)

    #Test for MLP and BM
    z = z_l2_mlp - z_l2_bm
    CI, p_value = paired_t_test(z)
    print(f"The statistical test for the MLP and BM has a CI of {CI} and a p-value of {p_value}")

    #Test for MLP an LM
    z = z_l2_mlp - z_l2_lm
    CI, p_value = paired_t_test(z)
    print(f"The statistical test for the MLP and LM has a CI of {CI} and a p-value of {p_value}")

    #Test for LM and BM
    z = z_l2_lm - z_l2_bm
    CI, p_value = paired_t_test(z)
    print(f"The statistical test for the LM and BM has a CI of {CI} and a p-value of {p_value}")
            
            
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
    
    #regression_b(X_regr, y_regr)
        
    
    
    
if __name__ == "__main__":
    main()
    
    