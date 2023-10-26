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

from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet, LogisticRegression
from sklearn.neural_network import MLPRegressor
import torch

from p2_utils import Standardize, InnerCVDataCollection, OuterCVDataCollection, get_best_parameters

from scipy.stats import norm
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

class BaselineClassification():
    def __init__(self):
        None

    def fit(self, X, y):
        self.prediction = np.argmax(np.bincount(y))

    def predict(self, X):
        return np.ravel(self.prediction*np.ones((X.shape[0],1)))
    
class ANNClassification():
    def __init__(self, n_hidden_units, n_input_units, n_output_units, n_replicates=1, max_iter=10**4):
        self.model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(n_input_units, n_hidden_units), 
                    #torch.nn.Tanh(), 
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units, n_output_units),
                    torch.nn.Softmax(dim=1) 
                    )
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.n_replicates = n_replicates
        self.max_iter = max_iter

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        tolerance = 10**-9
        best_final_loss = 10**100
        for r in range(self.n_replicates):
            print('\n\tReplicate: {}/{}'.format(r+1, self.n_replicates))
            net = self.model()

            torch.nn.init.xavier_uniform_(net[0].weight)
            torch.nn.init.xavier_uniform_(net[2].weight)

            optimizer = torch.optim.Adam(net.parameters())

            print('\t\t{}\t{}\t\t{}'.format('Iter', 'Loss','Rel. loss'))
            learning_curve = []
            old_loss = 1e6
            for i in range(int(self.max_iter)):
                y_est = net(X)
                loss = self.loss_fct(y_est, y)
                loss_value = loss.data.numpy()
                learning_curve.append(loss_value)

                p_delta_loss = np.abs(loss_value-old_loss)/old_loss
                if p_delta_loss < tolerance:
                    break
                old_loss = loss_value

                if (i != 0) & ((i+1) % 1000 == 0):
                    print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
                    print(print_str)

                optimizer.zero_grad(); loss.backward(); optimizer.step()

            print('\t\tFinal loss:')
            print('\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss))

            if loss_value < best_final_loss:
                self.net = net
                self.loss = loss_value
                self.learning_curve = learning_curve

    def predict(self, X):
        X = torch.Tensor(X)

        y = self.net(X)
        return (torch.max(y, dim=1)[1]).data.numpy()
    
    def get_learning_curve(self):
        return self.learning_curve

def one_out_of_k_encoding(y):
    new_y = np.zeros((y.shape[0], len(np.unique(y))))
    for i in range(y.shape[0]):
        new_y[i,y[i]] = 1

    return new_y

def train_bm(collector, param_list, X_train, y_train, X_val, y_val):
    p=0
    for param in param_list:
        #model needs to be adapted once seen in the lecture
        #training of ann model
        model = BaselineClassification()
        
        y_train_pred, y_val_pred, error_train, error_val = train_class(model, X_train, y_train, X_val, y_val)
                
        collector.set_perf(error_val, 0)
                
        p=p+1
    
    return collector

def train_lm(collector, param_list, X_train, y_train, X_val, y_val):
    p=0
    for param in param_list:
        #model needs to be adapted once seen in the lecture
        #training of ann model
        model = LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, random_state=44, penalty='l2', C=1/param, max_iter=1**3)
        
        y_train_pred, y_val_pred, error_train, error_val = train_class(model, X_train, y_train, X_val, y_val)
                
        collector.set_perf(error_val, 0)
                
        p=p+1
    
    return collector

def train_ann(collector, param_list, n_classes, X_train, y_train, X_val, y_val):
    p=0
    for param in param_list:
        #model needs to be adapted once seen in the lecture
        #training of ann model
        model = ANNClassification(n_hidden_units=param, n_input_units=X_train.shape[1], n_output_units=n_classes, n_replicates=2, max_iter=1.5*10**4)
        
        y_train_pred, y_val_pred, error_train, error_val = train_class(model, X_train, y_train, X_val, y_val)
                
        collector.set_perf(error_val, p)
                
        p=p+1
    
    return collector

def train_class(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
                
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    error_train = np.sum((y_train_pred != y_train))/y_train.shape[0]
    error_val = np.sum((y_val_pred != y_val))/y_val.shape[0]            
    #error_train = np.count_nonzero(np.sum(np.abs(y_train-y_train_pred), axis=1))/y_train.shape[0]
    #error_val = np.count_nonzero(np.sum(np.abs(y_val-y_val_pred),axis=1))/y_val.shape[0]
    return y_train_pred, y_val_pred, error_train, error_val

def mcnemars_test(cA, cB):
    cA, cB = np.concatenate(cA), np.concatenate(cB)
    alpha=0.05

    n11 = np.sum(np.multiply(cA,cB))
    n12 = np.sum(np.multiply(cA,np.ones(cB.shape)-cB))
    n21 = np.sum(np.multiply(np.ones(cA.shape)-cA,cB))
    n22 = np.sum(np.multiply(np.ones(cA.shape)-cA,np.ones(cB.shape)-cB))
    n = n11+n12+n21+n22
    theta_est = (n12-n21)/n
    Q = (n**2 * (n+1) * (theta_est+1) * (1-theta_est))/(n*(n12+n21)-((n12-n21)**2))
    f = (theta_est+1)*(Q-1)/2
    g = (1-theta_est)*(Q-1)/2

    CI = tuple(i * 2 -1 for i in st.beta.interval(1-alpha, a=f, b=g))

    p = 2*st.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)

    return CI, p

def classification(X_class, y_class):
    #Classification
    n_classes = np.max(y_class)+1

    #initialize CV
    outer_K = 10
    outer_CV = KFold(n_splits=outer_K, shuffle=True, random_state=44)
    
    inner_K = 10
    inner_CV = KFold(n_splits=inner_K, shuffle=True, random_state=44)

    #create list of arrays to store all inner CV data
    inner_CV_list = []
    
    #create outer collector to store outer CV data
    outer_collector = OuterCVDataCollection(outer_K)

    #define ANN parameters
    ann_param = [1, 10, 20, 50]
    
    #define lineare regression parameters
    lm_param = [10**-9, 10**-6, 10**-3, 1, 1000]

    #initialize loss lists
    c_bm = []
    c_lm = []
    c_ann = []

    #list to collect coefficents of best multinomial regression model
    coef_list = []

    j=0
    for par_idx, test_idx in outer_CV.split(X_class):
        print(f"Outer Crossvalidation Loop {j+1}")

         #initialization of training ad test data for the split
        X_par, y_par = X_class.values[par_idx,:], y_class.values[par_idx]
        X_test, y_test = X_class.values[test_idx,:], y_class.values[test_idx]
        
        #inner CV loop
        i=0
        for train_idx, val_idx in inner_CV.split(X_par):
            print(f"Inner Crossvalidation Loop {i+1}")
            #initialization of training ad test data for the split
            X_train, y_train = X_par[train_idx,:], y_par[train_idx]
            X_val, y_val = X_par[val_idx,:], y_par[val_idx]
            
            #shall this be done for every CV or just once at the begining?
            #standardization of the training data
            standardizer = Standardize()
            X_train = standardizer.fit(X_train)
            X_val = standardizer.transform(X_val)

            inner_collector_bm = InnerCVDataCollection(1)
            inner_collector_ann = InnerCVDataCollection(len(ann_param))
            inner_collector_lm = InnerCVDataCollection(len(lm_param))

            inner_collector_bm = train_bm(inner_collector_bm, [1], X_train, y_train, X_val, y_val)

            inner_collector_ann = train_ann(inner_collector_ann, ann_param, n_classes, X_train, y_train, X_val, y_val)

            inner_collector_lm = train_lm(inner_collector_lm, lm_param, X_train, y_train, X_val, y_val)
             
            inner_CV_list.append([inner_collector_bm, inner_collector_lm, inner_collector_ann])

            i = i+1

        #standardize the data
        standardizer = Standardize()
        X_par = standardizer.fit(X_par)
        X_test = standardizer.transform(X_test)
        
        #train best basemodel of inner CV
        bm = BaselineClassification()
        bm.fit(X_par,y_par)
        
        y_par_pred_bm = bm.predict(X_par)
        y_test_pred_bm = bm.predict(X_test)

        c_bm.append((y_test_pred_bm == y_test))
        
        err_par_bm = np.sum((y_par_pred_bm != y_par))/y_par.shape[0]
        err_test_bm = np.sum((y_test_pred_bm != y_test))/y_test.shape[0]
        
        outer_collector.set_bm_perf(err_test_bm, j)

        #train best ANN of inner CV
        best_param_idx, best_param = get_best_parameters(inner_CV_list, ann_param, 2)
        
        ann = ANNClassification(n_hidden_units=best_param, n_input_units=X_par.shape[1], n_output_units = n_classes, n_replicates=2)
        ann.fit(X_par,y_par)
        
        y_par_pred_ann = ann.predict(X_par)
        y_test_pred_ann = ann.predict(X_test)

        c_ann.append((y_test_pred_ann == y_test))
        
        err_par_ann = np.sum((y_par_pred_ann != y_par))/y_par.shape[0]
        err_test_ann = np.sum((y_test_pred_ann != y_test))/y_test.shape[0]
        
        outer_collector.set_ann_perf(err_test_ann, j)
        outer_collector.set_ann_param_idx(best_param_idx, j)
        
        #train best Linear Model of inner CV
        best_param_idx, best_param = get_best_parameters(inner_CV_list, lm_param, 1)
        
        lm = LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, random_state=44, penalty='l2', C=1/best_param, max_iter=1**3)
        lm.fit(X_par,y_par)
        
        y_par_pred_lm = lm.predict(X_par)
        y_test_pred_lm = lm.predict(X_test)

        c_lm.append((y_test_pred_lm == y_test))
        
        err_par_lm = np.sum((y_par_pred_lm != y_par))/y_par.shape[0]
        err_test_lm = np.sum((y_test_pred_lm != y_test))/y_test.shape[0]
        
        outer_collector.set_lm_perf(err_test_lm, j)
        outer_collector.set_lm_param_idx(best_param_idx, j)
        coef_list.append(np.concatenate((lm.coef_,np.reshape(lm.intercept_,(lm.intercept_.shape[0],1))),axis=1))

        j = j+1

    #create summarization table for the outer CV loop
    table_b_df = pd.DataFrame(columns=["ANN_param", "ANN_error", "LM_param", "LM_error", "BM_error"], index=range(outer_K))       
    for r in range(outer_K):
        table_b_df["ANN_param"][r] = ann_param[int(outer_collector.get_ann_param_idx()[0,r])]
        table_b_df["ANN_error"][r] = outer_collector.get_ann_perf()[0,r]
        table_b_df["LM_param"][r] = lm_param[int(outer_collector.get_lm_param_idx()[0,r])]
        table_b_df["LM_error"][r] = outer_collector.get_lm_perf()[0,r]
        table_b_df["BM_error"][r] = outer_collector.get_bm_perf()[0,r]
    
    with open(os.path.join(os.getcwd(),'classification_table.txt'), 'w') as f:
        f.write(table_b_df.to_string())

    #Test for ANN and BM
    CI, p_value = mcnemars_test(c_ann, c_bm)
    print(f"The statistical test for the ANN and BM has a CI of {CI} and a p-value of {p_value}")

    #Test for ANN an LM
    CI, p_value = mcnemars_test(c_ann, c_lm)
    print(f"The statistical test for the ANN and LM has a CI of {CI} and a p-value of {p_value}")

    #Test for LM and BM
    CI, p_value = mcnemars_test(c_lm, c_bm)
    print(f"The statistical test for the LM and BM has a CI of {CI} and a p-value of {p_value}") 

    best_lm_idx = np.argmin(outer_collector.get_lm_perf())
    seasons = ["winter", "spring", "summer", "fall"]
    for i in range(coef_list[best_lm_idx].shape[0]):
        plt.figure()
        plt.barh(np.concatenate((np.asarray(X_class.columns),["bias"]),0), coef_list[best_lm_idx][i,:].flatten())
        plt.title(f'Coefficients of Best Linear Regression Model for class {seasons[i]}')
        plt.xlabel('coefficient')
        plt.ylabel('feature')
        plt.show()

    print("classification task done")
       