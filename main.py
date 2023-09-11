# -*- coding: utf-8 -*-
# project 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats as st

def main():
    ### load data ###
    data_path = os.path.join(os.getcwd(),"LAozone.csv")
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
    
    ### inspect data ###
    # finding NaN values
    missing_idx = np.isnan(data)
    missing_counter = np.sum(missing_idx, 0) > 0
    
    plt.figure().set_figheight(1)
    plt.plot(missing_counter.index,missing_counter.values.reshape(len(missing_counter),1), 'bo')
    plt.title("Missing Values (NaN) in Features")
    plt.xlabel("features")
    plt.ylabel("NaNs present")
    plt.ylim(-0.4, 1.4)
    plt.yticks([0,1],["FALSE", "TRUE"])
    plt.show()
    
    # basic statistical metrics
    stat_df = pd.DataFrame(columns=data.columns, index=("max", "min", "mean", "median", "std", "std (N-1)", "var", "var (N-1)", "mode"))
    for column in data.columns:
        stat_df[column]["mean"] = np.mean(data[column])
        stat_df[column]["median"] = np.median(data[column])
        stat_df[column]["min"] = np.min(data[column])
        stat_df[column]["max"] = np.max(data[column])
        stat_df[column]["std"] = np.std(data[column])
        stat_df[column]["std (N-1)"] = np.std(data[column], ddof=1)
        stat_df[column]["var"] = np.var(data[column])
        stat_df[column]["var (N-1)"] = np.var(data[column], ddof=1)
        stat_df[column]["mode"] = st.mode(data[column])
    
    with open(os.path.join(os.getcwd(),"data_measures.txt"), 'w') as f:
        f.write(stat_df.to_string())
    
    # plot boxplots/distribution of features
    plt.boxplot(data, labels=data.columns)
    plt.title("Boxplots of all Features")
    plt.xlabel("features")
    plt.ylabel("data values")
    plt.show()
    
    n_bins = 25
    fig, ax = plt.subplots(2, int(np.ceil(len(data.columns)/2)))
    plt.figure().set_figheight(10)
    plt.figure().set_figwidth(20)
    fig.tight_layout()
    for col_id in range(len(data.columns)):
        if col_id < int(np.ceil(len(data.columns)/2)):
            ax[0,col_id].hist(data.iloc[:,col_id], bins=n_bins)
            ax[0,col_id].set_title(data.columns[col_id])
        if col_id >= int(np.ceil(len(data.columns)/2)):
            ax[1,col_id-int(np.ceil(len(data.columns)/2))].hist(data.iloc[:,col_id], bins=n_bins)
            ax[1,col_id-int(np.ceil(len(data.columns)/2))].set_title(data.columns[col_id])
    plt.show()
    
    # show correlations
    
    sns.heatmap(data.corr(), cmap="RdBu")
    
    ### transform data ###
    
    # one hot encoding (if needed)
    
    # standardize
    
    
    
    ### PCA ###


if __name__ == "__main__":
    main()
