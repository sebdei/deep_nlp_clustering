import pandas as pd
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

#os.chdir("C:\\Users\\k_lim002\\Desktop\\Seminar")


matrix = pd.read_pickle("Word_Matrices.pkl")


x_train = np.array([])
for i in range(len(matrix)):
    x_train = np.append( x_train, matrix.iloc[i,3])
    
x_train = x_train.reshape((7299, 1))



components = 10
explained var = []
for components in range (1 ,100 ,5) :
    pca = PCA( n_components=components )
    pca.fit (matrix["WordMatrix"])
    explained_var.append(pca.explained_variance_ratio.sum())

plt.plot(range(1,100,5),explained var ,"ro") plt.xlabel ("Number of Components")
          plt.ylabel ("Proportion of Explained Variance")


components = 60
palette = np.array(sns.color palette(”hls”, 120))
lsa = TruncatedSVD(n components=components) lsa . fit (dtm)
lsa dtm = lsa . transform (dtm)
