# -*- coding: utf-8 -*-
# project 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats as st
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler

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
    stat_df = pd.DataFrame(columns=data.columns, index=("max", "min", "mean", "median", "std", "std (N-1)", "var", "var (N-1)", "mode", "25%-percentile", "75%-percentile"))
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
        stat_df[column]["25%-percentile"] = np.quantile(data[column], 0.25)
        stat_df[column]["75%-percentile"] = np.quantile(data[column], 0.75)

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
    
    data_pca = data.drop(["doy", "season"], axis=1)

    scaler = StandardScaler()
    scaler.fit(data_pca)
    data_pca_scaled = scaler.transform(data_pca)

    ### PCA ###

    U,S,V = svd(data_pca_scaled, full_matrices=False)

    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum()

    plt.figure()
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative'])
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
    #Z = array(Z)
    for c in range(len(sorted(set(data["season"])))):
        # select indices belonging to class c:
        class_mask = data["season"]==c
        plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
    plt.legend(["winter", "spring", "summer", "fall"])
    plt.xlabel('PC{0}'.format(i+1))
    plt.ylabel('PC{0}'.format(j+1))

    # Output result to screen
    plt.show()


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

    # Set up the plot
    plt.figure(figsize=(10, 6))
    colors = ['green', "red", "blue", "orange"]
    # Horizontal line for 1D representation
    plt.axhline(y=1, color='grey', linestyle='--', lw=0.5)

    # Plot data with different symbols for each season on the horizontal line
    for i, row in data.iterrows():
        print(data["season"][i])
        plt.scatter(row['temp'], 1, color=colors[data["season"][i]])

    # Set plot details
    plt.title("Temperature with Season Symbols")
    plt.xlabel("Temperature (°C)")
    plt.yticks([])  # Hide y-ticks as it's a 1D plot
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, axis='x')

    # Display the plot
    plt.tight_layout()
    plt.show()

    f = plt.figure()
    plt.title('NanoNose data')

    for c in range(4):
        # select indices belonging to class c:
        class_mask = data["season"] == c
        plt.plot(data["temp"][class_mask], data["vis"][class_mask], 'o', alpha=.3)

#    plt.legend(data["season"])
 #   plt.xlabel(data["temp"])
  #  plt.ylabel(data["vis"])
    plt.show()

if __name__ == "__main__":
    main()
