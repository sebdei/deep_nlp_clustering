import numpy as np
from keras.utils import plot_model
import pandas as pd
import os
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from keras.utils import plot_model
from CNN_Utility_Functions import 
from CNN_Utility_Functions import CNN_autoencoder_2D,recreate_input_matrix_2d, target_distribution
from clustering import ClusteringLayer, get_init_cluster_center
import keras.backend as K
from sklearn.cluster import KMeans


#os.chdir("/Volumes/Files/Onedrive/Masters/Study Materials/Third Semester/Seminar-Recent Trends in Deep Learning")
#os.chdir("/Users/kevin/Downloads")
n_data = 100

matrix = pd.read_pickle("Word_Matrices.pkl")
matrix = matrix[:n_data]

x_train = recreate_input_matrix_2d(matrix, 3,n_data, 24, 300)

autoencoder, encoder  = CNN_autoencoder_2D(x_train, (7,76), (8,100))




n_clusters = 3
maxiter = 10
update_interval = 2
tol=1e-3
index = 0
batch_size = 20

latent_features = encoder.predict(x_train)
init_cluster_centers = get_init_cluster_center(n_clusters, latent_features)

# Define DCEC model
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=autoencoder.input, outputs=[clustering_layer, autoencoder.output])
model.compile(loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam')
model.get_layer(name='clustering').set_weights([init_cluster_centers])
plot_model(model, show_shapes=True, to_file='clustering.png')

kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(x_train))
y_pred_last = np.copy(y_pred)

for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q, _ = model.predict(x_train)
        p = target_distribution(q)  

        y_pred = q.argmax(1)
  
        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    if (index + 1) * batch_size > x_train.shape[0]:
        loss = model.train_on_batch(x=x_train[index * batch_size::],
                                            y=[p[index * batch_size::], x_train[index * batch_size::]])
        index = 0
    else:
        loss = model.train_on_batch(x=x_train[index * batch_size:(index + 1) * batch_size],
                                            y=[p[index * batch_size:(index + 1) * batch_size],
                                            x_train[index * batch_size:(index + 1) * batch_size]])
        index += 1
    ite += 1


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x =data[:,0]
y =data[:,1]
ax.scatter(x, y, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')

plt.show()

# Implementation  of Deep Clustering with Convolutional Autoencoders Xifeng Guo1, Xinwang Liu1, En Zhu1, and Jianping Yin2

