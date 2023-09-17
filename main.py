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

from scipy.stats import norm
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)



### inspect data ###
# finding NaN or no values
# looking for duplicates
def inspect_data(data):
    print("Is there missing Data?: ", data.isnull().sum().sum())
    print("Is the duplicated data?:", data.duplicated().sum())

    # count, mean, std, min, 25, 50(median), 75, max
    #with open(os.path.join(os.getcwd(), "data_measures.txt"), 'w') as f:
    #    f.write(data.describe().round(2).to_string())


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

    with open(os.path.join(os.getcwd(), "data_measures.txt"), 'w') as f:
        f.write(stat_df.to_string())


# Data Visualisation
def data_visualisation(data):


    # plot boxplots/distribution of features
    plt.figure(figsize=(10, 8))
    plt.boxplot((data - data.mean()) / data.std() , labels=data.columns)
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



    # show correlations
    plt.figure(figsize=(10,8))
    sns.heatmap(data.corr(), cmap="RdBu")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Correlation Heat Map")
    plt.tight_layout
    plt.show()

    #with open(os.path.join(os.getcwd(), "data_measures.txt"), 'w') as f:
    #    f.write(data.corr().to_string())

def pca(data):
    ### transform data ###
    # one hot encoding (if needed)

    # standardize

    data_pca = data.drop(["doy", "season"], axis=1)

    scaler = StandardScaler()
    scaler.fit(data_pca)
    data_pca_scaled = scaler.transform(data_pca)

    ### PCA ###
    U, S, V = svd(data_pca_scaled, full_matrices=False)

    # Compute variance explained by principal components
    rho = (S * S) / (S * S).sum()



    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, 'x-')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual', 'Cumulative'])
    plt.grid()
    plt.show()

    V_real = V.T
    Z = data_pca_scaled @ V_real

    # Indices of the principal components to be plotted
    i = 0
    j = 1

    # Plot PCA of the data
    plt.figure()
    plt.title('Los Angeles Ozone: PCA')
    # Z = array(Z)
    for c in range(len(sorted(set(data["season"])))):
        # select indices belonging to class c:
        class_mask = data["season"] == c
        plt.plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
    plt.legend(["winter", "spring", "summer", "fall"])
    plt.xlabel('PC{0}'.format(i + 1))
    plt.ylabel('PC{0}'.format(j + 1))

    # Output result to screen
    plt.show()



def pca_analysis(data):
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

    plt.legend(data["season"])
    #plt.xlabel(data["temp"])
    #plt.ylabel(data["vis"])
    plt.show()


    # Show projection of data onto principal components
    mean = data.mean()
    std = data.std()
    data_normalized = (data - mean) / std

    pca = PCA()
    pca.fit(data_normalized.drop(["doy", "season"]))
    num_components = 5

    fig, ax = plt.subplots(figsize=(14, 8))

    for i in range(num_components):
        ax.plot(data.columns, pca.components_[i], label=f'Component {i + 1}', marker='o')

    for i in range(num_components):
        print(pca.components_[i])

    ax.set_xticks(data.columns)
    ax.set_xticklabels(data.columns, rotation=45)
    ax.set_ylabel('Loading')
    ax.set_title('PCA Component Loadings for Each Feature')
    ax.legend()
    ax.grid(True)
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

    #inspect_data(data)
    #data_visualisation(data)
    #pca(data)
    pca_analysis(data)








if __name__ == "__main__":
    main()
