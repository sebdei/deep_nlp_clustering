import pandas as pd
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer

# Execute this for Word embedding
matrix = pd.read_pickle("Word_Matrices_60.pkl")

Flatten_Data = np.array([matrix['WordMatrix'][i] for i in range(len(matrix['WordMatrix']))])
Flatten_Data = Flatten_Data.reshape(7135, 18000)
Labels = matrix['Rating'].values


# Execute this if TF-IDF
dataset = pd.read_csv("bbc.csv")
v = TfidfVectorizer()
Flatten_Data = v.fit_transform(dataset['text']).toarray()
Labels = dataset['label'].values

# Check number of components
maxcomponent = 1000
explained_var = []
for components in range(1, maxcomponent, 5):
    pca = PCA(n_components=components)
    pca.fit(Flatten_Data)
    explained_var.append(pca.explained_variance_ratio_.sum())


# Plot the explained variance
plt.plot(range(1, maxcomponent, 5), explained_var, "ro")
plt.xlabel("Number of Components")
plt.ylabel("Proportion of Explained Variance")
plt.show()


# Split train test
stratSplit = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=42)
for train_index, test_index in stratSplit.split(Flatten_Data, Labels):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = Flatten_Data[train_index], Flatten_Data[test_index]
    y_train, y_test = Labels[train_index], Labels[test_index]


# Raw Kmeans
model = KMeans(n_clusters=5, max_iter=100)
clustered = model.fit(X_train)
labels_pred = model.predict(X_test)
metrics.fowlkes_mallows_score(y_test, labels_pred)


# Chose 80 components
pca_train = PCA(n_components=1000)
pca_train.fit(X_train)
pca_test = pca.transform(X_test)
pca_train = pca.transform(X_train)


# Clusters based on stars for PCA
model = KMeans(n_clusters=5, max_iter=100)
clustered = model.fit(pca_train)
labels_pred = model.predict(pca_test)

metrics.fowlkes_mallows_score(y_test, labels_pred)
metrics.homogeneity_score(y_test, labels_pred)


# LSA
lsa_train = TruncatedSVD(n_components=components)
lsa_train.fit(X_train)
lsa_test = lsa_train.transform(X_test)
lsa_train = lsa_train.transform(X_train)


# Clusters based on stars for LSA
model = KMeans(n_clusters=5, max_iter=100)
clustered = model.fit(lsa_train)
labels_pred = model.predict(lsa_test)

metrics.fowlkes_mallows_score(y_test, labels_pred)
metrics.homogeneity_score(y_test, labels_pred)




