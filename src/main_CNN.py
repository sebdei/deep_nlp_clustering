import numpy as np
from keras.utils import plot_model
import pandas as pd
import os
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from keras.utils import plot_model
from CNN_Utility_Functions import CNN_autoencoder_2D,recreate_input_matrix_2d, target_distribution
from clustering import ClusteringLayer
import keras.backend as K
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit

#os.chdir("/Volumes/Files/Onedrive/Masters/Study Materials/Third Semester/Seminar-Recent Trends in Deep Learning")
#os.chdir("/Users/kevin/Downloads")
#os.chdir("C:\\Users\\k_lim002\\Desktop\\Seminar")



source = pd.read_pickle("Word_Matrices_small.pkl")
n_data = len(source)

matrix = recreate_input_matrix_2d(source, 3,n_data, 30, 300)
rating = source['Rating'].values

stratSplit = StratifiedShuffleSplit(n_splits=5,test_size=0.4, random_state=42)
for train_index, test_index in stratSplit.split(matrix, rating):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = matrix[train_index], matrix[test_index]
    y_train, y_test = rating[train_index], rating[test_index]

autoencoder, encoder  = CNN_autoencoder_2D(X_train, (2,2), (10,100))


n_clusters = 5
maxiter = 5
update_interval = 3
tol=1e-3
index = 0
batch_size = 200

# Implementation  of Deep Clustering with Convolutional Autoencoders Xifeng Guo1, Xinwang Liu1, En Zhu1, and Jianping Yin2

# Define DCEC model
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(X_train))


clustering_layer = ClusteringLayer(n_clusters,weights=[kmeans.cluster_centers_], name='clustering')(encoder.output)
model = Model(inputs=autoencoder.input, outputs=[clustering_layer, autoencoder.output])
model.compile(loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam')


plot_model(model, show_shapes=True, to_file='clustering.png')



y_pred_last = np.copy(y_pred)
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q, _ = model.predict(X_train)
        p = target_distribution(q)  

        y_pred = q.argmax(1)
  
        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    if (index + 1) * batch_size > X_train.shape[0]:
        loss = model.train_on_batch(x=X_train[index * batch_size::],
                                            y=[p[index * batch_size::], X_train[index * batch_size::]])
        index = 0
    else:
        loss = model.train_on_batch(x=X_train[index * batch_size:(index + 1) * batch_size],
                                            y=[p[index * batch_size:(index + 1) * batch_size],
                                            X_train[index * batch_size:(index + 1) * batch_size]])
        index += 1
    ite += 1

model.save_weights("model.h5")

y_pred,_ =  model.predict(X_test)
y_pred=y_pred.argmax(1)


metrics.fowlkes_mallows_score(y_test, y_pred)  #0.3652808738375953
metrics.homogeneity_score(y_test, y_pred)  #0.019551600144295616


