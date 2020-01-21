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
from sklearn.feature_extraction.text import CountVectorizer
import text_provider
import model_provider

#os.chdir("C:\\Users\\k_lim002\\Desktop\\Seminar")

# Open files
data, label = text_provider.provide_amazon_sequence_list()
#data, label = text_provider.provide_bbc_sequence_list()


#Create Bag of Words with no dimensionality reduction
data_BoW = pd.Series([' '.join(data) for data in data])
vectorizer = CountVectorizer()
data_BoW = vectorizer.fit_transform(data_BoW).toarray()

vectorizer = CountVectorizer()
data_BoW = vectorizer.fit_transform(data).toarray()

#Split train test 
stratSplit = StratifiedShuffleSplit(n_splits=5,test_size=0.2, random_state=42)
for train_index, test_index in stratSplit.split(data_BoW, label):
    X_train, X_test = data_BoW[train_index], data_BoW[test_index]
    y_train, y_test = label[train_index], label[test_index]

#Kmeans on Bag-of-Words with no dimensionality reduction
model = KMeans(n_clusters=5,max_iter=100) 
clustered = model.fit(X_train)
labels_pred = model.predict(X_test)
metrics.fowlkes_mallows_score(y_test, labels_pred) 
print(metrics.fowlkes_mallows_score(y_test, labels_pred) )

#PCA on Bag-of-Words with no dimensionality reduction
#Check number of components
maxcomponent=len(data_BoW)
explained_var = []
for components in range (1 ,maxcomponent ,10) :
    pca = PCA( n_components=components )
    pca.fit (data_BoW)
    explained_var.append(pca.explained_variance_ratio_.sum())
    if pca.explained_variance_ratio_.sum() > 0.6:
        break

pca_train = PCA( n_components=260)
pca_train.fit(X_train)
pca_test = pca.transform(X_test)
pca_train = pca.transform(X_train)
model = KMeans(n_clusters=5,max_iter=100) 
clustered = model.fit(pca_train)
labels_pred = model.predict(pca_test)
metrics.fowlkes_mallows_score(y_test, labels_pred)  


#Create  TF-IDF with no dimensionality reduction
data_Tfidf = pd.Series([' '.join(data) for data in data])
vectorizer =TfidfVectorizer()
data_Tfidf = vectorizer.fit_transform(data_Tfidf).toarray()

vectorizer =TfidfVectorizer()
data_Tfidf = vectorizer.fit_transform(data).toarray()

#Split train test 
stratSplit = StratifiedShuffleSplit(n_splits=5,test_size=0.2, random_state=42)
for train_index, test_index in stratSplit.split(data_Tfidf, label):
    X_train, X_test = data_Tfidf[train_index], data_Tfidf[test_index]
    y_train, y_test = label[train_index], label[test_index]

#Kmeans on TF-IDF with no dimensionality reduction
model = KMeans(n_clusters=5,max_iter=100) 
clustered = model.fit(X_train)
labels_pred = model.predict(X_test)
metrics.fowlkes_mallows_score(y_test, labels_pred) 

#PCA on Bag-of-Words with no dimensionality reduction
#Check number of components
maxcomponent=len(data_Tfidf)
explained_var = []
for components in range (620 ,maxcomponent ,10) :
    pca = PCA( n_components=components )
    pca.fit (data_Tfidf)
    explained_var.append(pca.explained_variance_ratio_.sum())
    print(pca.explained_variance_ratio_.sum())
    if pca.explained_variance_ratio_.sum() > 0.6:
        break

pca_train = PCA( n_components=590)
pca_train.fit(X_train)
pca_test = pca.transform(X_test)
pca_train = pca.transform(X_train)
model = KMeans(n_clusters=5,max_iter=100) 
clustered = model.fit(pca_train)
labels_pred = model.predict(pca_test)
metrics.fowlkes_mallows_score(y_test, labels_pred)  


#LSA
data_LSA = pd.Series([' '.join(data) for data in data])
vectorizer =TfidfVectorizer()
data_LSA = vectorizer.fit_transform(data_LSA).toarray()

vectorizer =TfidfVectorizer()
data_LSA = vectorizer.fit_transform(data).toarray()

#Split train test 
stratSplit = StratifiedShuffleSplit(n_splits=5,test_size=0.2, random_state=42)
for train_index, test_index in stratSplit.split(data_LSA, label):
    X_train, X_test = data_LSA[train_index], data_LSA[test_index]
    y_train, y_test = label[train_index], label[test_index]

maxcomponent=len(data_LSA)
explained_var = []
for components in range (500 ,maxcomponent ,10) :
    lsa = TruncatedSVD(n_components=components) 
    lsa.fit (data_LSA)
    explained_var.append(lsa.explained_variance_ratio_.sum())
    print(lsa.explained_variance_ratio_.sum())
    if lsa.explained_variance_ratio_.sum() > 0.6:
        break

lsa_train = TruncatedSVD(n_components=620) 
lsa_train.fit(X_train)
lsa_test = lsa_train.transform(X_test)
lsa_train = lsa_train.transform(X_train)

#Clusters based on stars for LSA
model = KMeans(n_clusters=5,max_iter=100) 
clustered = model.fit(lsa_train)
labels_pred = model.predict(lsa_test)

metrics.fowlkes_mallows_score(y_test, labels_pred)  
metrics.homogeneity_score(y_test, labels_pred) 


#Create FastText with no dimensionality reduction
def createFastTextArray(input, model, maxLen):
    counter = 0
    arr = np.zeros((maxLen,300))
    for i in range(0, len(input)-2):
        try:
            arr[i - counter] = model[input[i]]
        except:
            counter +=1
    return arr

def fasttextdata(data):
    model = model_provider.provide_fasttext_model()
    maxLen =  max([len(x) for x in data])
    data_fastext = np.array([createFastTextArray(x, model, maxLen) for x in data])
    return data_fastext

data_fastext = fasttextdata(data)
data_fastext = data_fastext.reshape(len(data_fastext),len(data_fastext[0])*len(data_fastext[0][0]))
#Split train test 
stratSplit = StratifiedShuffleSplit(n_splits=5,test_size=0.2, random_state=42)
for train_index, test_index in stratSplit.split(data_fastext, label):
    X_train, X_test = data_fastext[train_index], data_fastext[test_index]
    y_train, y_test = label[train_index], label[test_index]

model = KMeans(n_clusters=5,max_iter=100) 
clustered = model.fit(X_train)
labels_pred = model.predict(X_test)
metrics.fowlkes_mallows_score(y_test, labels_pred) 

maxcomponent=len(data_fastext)
explained_var = []

for components in range (380 ,maxcomponent ,10) :
    pca = PCA(n_components=components)
    pca.fit(data_fastext)
    explained_var.append(pca.explained_variance_ratio_.sum())
    print(pca.explained_variance_ratio_.sum())
    if pca.explained_variance_ratio_.sum() > 0.6:
        print(components)
        break

pca_train = PCA( n_components=380)
pca_train.fit(X_train)
pca_test = pca.transform(X_test)
pca_train = pca.transform(X_train)
model = KMeans(n_clusters=5,max_iter=100) 
clustered = model.fit(pca_train)
labels_pred = model.predict(pca_test)
metrics.fowlkes_mallows_score(y_test, labels_pred)  