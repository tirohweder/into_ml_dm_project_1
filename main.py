# -*- coding: utf-8 -*-
# project 1

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import norm
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)



### inspect data ###
# finding NaN or no values
# looking for duplicates
def inspect_data(data):
    # check for missing data
    print("Is there missing Data?: ", data.isnull().sum().sum())
    # check for duplicated data
    print("Is there duplicated data?:", data.duplicated().sum())

    # count, mean, std, min, 25, 50(median), 75, max
    #with open(os.path.join(os.getcwd(), "data_measures.txt"), 'w') as f:
    #    f.write(data.describe().round(2).to_string())

    # calculation of simple summary statistics
    stat_df = pd.DataFrame(columns=data.columns, index=(
        "mean", "std","var","min", "25%-percentile", "median",  "75%-percentile", "max", "std (N-1)",  "var (N-1)",
        "mode"))
    for column in data.columns:
        stat_df[column]["mean"] = round(np.mean(data[column]),2)
        stat_df[column]["median"] = round(np.median(data[column]),2)
        stat_df[column]["min"] = round(np.min(data[column]),2)
        stat_df[column]["max"] = round(np.max(data[column]),2)
        stat_df[column]["std"] = round(np.std(data[column]),2)
        stat_df[column]["std (N-1)"] = round(np.std(data[column], ddof=1),2)
        stat_df[column]["var"] = round(np.var(data[column]),2)
        stat_df[column]["var (N-1)"] = round(np.var(data[column], ddof=1),2)
        stat_df[column]["mode"] = st.mode(data[column])
        stat_df[column]["25%-percentile"] = round(np.quantile(data[column], 0.25),2)
        stat_df[column]["75%-percentile"] = round(np.quantile(data[column], 0.75),2)
    
    # write summary statistics to file
    with open(os.path.join(os.getcwd(), "data_measures.txt"), 'w') as f:
        f.write(stat_df.to_string())

# Data Visualisation
def data_visualisation(data):
    ### plot boxplots/distribution of features ###
    plt.figure(figsize=(10, 8))
    plt.boxplot((data - data.mean()) / data.std(ddof=1) , labels=data.columns)
    plt.title("Boxplots of all Features")
    plt.xlabel("Features")
    plt.ylabel("Data values")
    plt.xticks(rotation=90)
    plt.show()

    #n_bins = 25
    #fig, ax = plt.subplots(2, int(np.ceil(len(data.columns) / 2)))
    #plt.figure().set_figheight(10)
    #plt.figure().set_figwidth(20)
    #fig.tight_layout()
    #for col_id in range(len(data.columns)):
    #    if col_id < int(np.ceil(len(data.columns) / 2)):
    #        ax[0, col_id].hist(data.iloc[:, col_id], bins=n_bins)
    #        ax[0, col_id].set_title(data.columns[col_id])
    #    if col_id >= int(np.ceil(len(data.columns) / 2)):
    #        ax[1, col_id - int(np.ceil(len(data.columns) / 2))].hist(data.iloc[:, col_id], bins=n_bins)
    #        ax[1, col_id - int(np.ceil(len(data.columns) / 2))].set_title(data.columns[col_id])
    #plt.show()
    
    ### plot histogramms ###
    # Set up the figure size and grid layout
    plt.figure(figsize=(15, 12))
    sns.set_style("whitegrid")

    # Plot histograms for each column
    for i, column in enumerate(data.columns.drop("season"), 1):
        plt.subplot(3, 4, i)
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.tight_layout()
    plt.show()

    ### plot correlations ###
    plt.figure(figsize=(10,8))
    sns.heatmap(data.corr(), cmap="RdBu")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Correlation Heat Map")
    plt.tight_layout
    plt.show()
    print(data.corr())
    
    #calculate empirical covariance and derive empirical correlation
    cov_mat = np.cov(data, rowvar=False, ddof=1)
    print(cov_mat)
    cor_mat = np.zeros((data.shape[1],data.shape[1]))
    
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            cor_mat[i][j] = cov_mat[i][j]/(np.std(data.iloc[:,i],ddof=1)*np.std(data.iloc[:,j],ddof=1))
            
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(cor_mat, cmap="RdBu")
    plt.xticks(rotation=90)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    plt.yticks(rotation=0)
    plt.title("Empirical Correlation Heat Map")
    plt.tight_layout
    plt.show()

    #with open(os.path.join(os.getcwd(), "data_measures.txt"), 'w') as f:
    #    f.write(data.corr().to_string())
    
    ### plot scatter for temperature ###
    # Temp - IBH IBT
    # Season - Temp vis
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(data["temp"], data["ibh"], color='blue', label='temp vs ibh')
    ax[0].set_title('Temperature vs IBH')
    ax[0].set_xlabel('Temperature')
    ax[0].set_ylabel('IBH')

    ax[1].scatter(data["temp"], data["ibt"], color='red', label='temp vs ibt')
    ax[1].set_title('Temperature vs IBT')
    ax[1].set_xlabel('Temperature')
    ax[1].set_ylabel('IBT')
    plt.show()
    
    ### Mapping season to temperature ####
    # Set up the plot
    plt.figure(figsize=(10, 6))
    colors = ['green', "red", "blue", "orange"]
    plt.axhline(y=1, color='grey', linestyle='--', lw=0.5)
    for i, row in data.iterrows():
        plt.scatter(row['temp'], 1, color=colors[data["season"][i]])
    plt.title("Temperature with Season Symbols")
    plt.xlabel("Temperature (°C)")
    plt.yticks([])  # Hide y-ticks as it's a 1D plot
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, axis='x')
    plt.tight_layout()
    plt.show()
    
    ###  Mapping season to temperature and visibility ###
    for c in range(4):
        # select indices belonging to class c:
        class_mask = data["season"] == c
        plt.plot(data["temp"][class_mask], data["vis"][class_mask], 'o', alpha=.3)

    #plt.legend(data["season"])
    plt.legend(["winter", "spring", "summer", "fall"])
    #plt.xlabel(data["temp"])
    #plt.ylabel(data["vis"])
    plt.show()

def pca(data):
    ### transform data ###
    # standardize
    data_pca = data.drop(["doy", "season"], axis=1)
    
    mean = data_pca.mean()
    std = data_pca.std(ddof=1)
    data_pca_scaled = np.asarray((data_pca - mean) / std)

    ### PCA ###
    U, S, V = svd(data_pca_scaled, full_matrices=False)

    # Compute variance explained by principal components
    rho = (S * S) / (S * S).sum()

    threshold = 0.9

    ### plot explained variance ###
    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, 'x-', color='red')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-', color='blue')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.show()

    ### transform data onto pca components ###
    V_real = V.T
    Z = data_pca_scaled @ V_real
    
    ### Plot PCA projection ###
    # pca component indices
    pca_idx = [[0, 1], [1, 4]]
    for idx in pca_idx:
        plt.figure()
        plt.title('Los Angeles Ozone: PCA')
        # Z = array(Z)
        for c in range(len(sorted(set(data["season"])))):
            # select indices belonging to class c:
            class_mask = data["season"] == c
            plt.plot(Z[class_mask, idx[0]], Z[class_mask, idx[1]], 'o', alpha=.5)
        plt.legend(["winter", "spring", "summer", "fall"])
        plt.xlabel('PC{0}'.format(idx[0] + 1))
        plt.ylabel('PC{0}'.format(idx[1] + 1))
        plt.show()
    
    ### further analysis of most important pca components ###
    # number of pca components to be analysed further
    max_pca = 5
    
    # plot matrix scatter pca plot for max_pca components
    fig, ax = plt.subplots(max_pca, max_pca, figsize=(20, 10))
    plt.suptitle(f'Los Angeles Ozone: PCA for {max_pca} components')
    
    for i in range(max_pca):
        for j in range(max_pca):
            for c in range(len(sorted(set(data["season"])))):
                # select indices belonging to class c:
                class_mask = data["season"] == c
                ax[i][j].plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
            
            ax[i][j].set_xlabel('PC{0}'.format(i + 1))
            ax[i][j].set_ylabel('PC{0}'.format(j + 1))
    plt.legend(["winter", "spring", "summer", "fall"])
    plt.tight_layout()
    plt.show()
    
    ### plot for pca contribution ###
    fig, ax = plt.subplots(figsize=(14, 8))

    for i in range(max_pca):
        ax.plot(data_pca.columns, V_real[:,i], label=f'Component {i + 1}', marker='o')

    for i in range(max_pca):
        print(V_real[:,i])
    
    ax.set_xticks(data_pca.columns)
    ax.set_xticklabels(data_pca.columns, rotation=45)
    ax.set_ylabel('Loading')
    ax.set_title('PCA Component Loadings for Each Feature')
    ax.grid(True)
    plt.show()
    
    ### pca heatmap ###
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(V_real[:,0:max_pca], cmap="RdBu")
    ax.legend()
    plt.colorbar(im)
    ax.set_yticks(np.arange(len(data_pca.columns)), labels=data_pca.columns)
    ax.set_xticks(np.arange(max_pca), labels=np.arange(max_pca)+1)
    ax.set_ylabel('Feature')
    ax.set_xlabel('PCA component')
    ax.set_title('PCA Component Loadings for Each Feature')
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

    inspect_data(data)
    data_visualisation(data)
    pca(data)
    
    # train the first classifiers
    
    #data_Y = data["season"].copy()
    
    #data_X = data.drop(["doy", "season"], axis=1).copy()
    
    #mean = data_X.mean()
    #std = data_X.std(ddof=1)
    #data_X = np.asarray((data_X - mean) / std)
    
    #X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size = 0.2, random_state=5, shuffle=True)

    #KNN = KNeighborsClassifier(n_neighbors = 10)
    #KNN.fit(X_train, y_train)
    
    #print(KNN.score(X_test,y_test))
    
    #DT = DecisionTreeClassifier()
    #DT.fit(X_train,y_train)
    
    #print(DT.score(X_test,y_test))
    
    #RF = RandomForestClassifier()
    #RF.fit(X_train,y_train)
    
    #print(RF.score(X_test,y_test))


if __name__ == "__main__":
    main()
