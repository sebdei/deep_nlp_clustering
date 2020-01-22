import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import text_provider 
import preprocess 
import CNN_Utility_Functions
import clustering_utils 
from keras.models import Model
import tensorflow as tf

'''
#BBC Data
n_clusters = 5
maxiter = 1120
update_interval = 56
tol=1e-3
index = 0
batch_size = 32
dimensions = 300

'''

#Amazon Data
n_clusters = 5
maxiter = 3580
update_interval = 179
tol=1e-3
index = 0
batch_size = 32
dimensions = 300


#data, label = text_provider.provide_bbc_sequence_list()
data, label = text_provider.provide_amazon_sequence_list()

embedding_matrix, x_train, x_test,y_train, y_test = preprocess.preprocess_word_embedding_fasttext(data, label)

vocab_size = len(embedding_matrix)
feature_dimension_size = len(embedding_matrix[0])
max_sequence_length = len( x_train[0])

autoencoder, encoder  = CNN_Utility_Functions.CNN_autoencoder_2D_em(x_train, (300,4),vocab_size,feature_dimension_size,max_sequence_length,embedding_matrix, "BBC")
#autoencoder, encoder  = CNN_Utility_Functions.CNN_autoencoder_2D_em(x_train, (300,4),vocab_size,feature_dimension_size,max_sequence_length,embedding_matrix, "Amazon")

autoencoder.save_weights("auto_encoder.h5")
encoder.save_weights("encoder.h5")

kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(x_train))


clustering_layer = clustering_utils.ClusteringLayer(n_clusters,weights=[kmeans.cluster_centers_], name='clustering')(encoder.output)
model = Model(inputs=autoencoder.input, outputs=[clustering_layer, autoencoder.output])
model.compile(loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam')
#model.compile(loss=['kld', 'cosine_proximity'], loss_weights=[1, 1], optimizer='adam')

y_pred_last = np.copy(y_pred)
expected_autoencoder_output = np.array([[embedding_matrix[word_index] for word_index in encoded_sequence] for encoded_sequence in x_train])
cnn_score = np.array([])

for ite in range(int(maxiter)):
    print("Interation: "+str(ite))
    if ite % update_interval == 0:
        q, _ = model.predict(x_train)
        p = clustering_utils.target_distribution(q)  

        y_pred = q.argmax(1)
        
        #y_pred_iter,_ =  model.predict(x_test)
        #y_pred_iter= y_pred_iter.argmax(1)
        #cnn_results=metrics.fowlkes_mallows_score(y_test,  y_pred_iter) 
        #cnn_score= np.append(cnn_score, cnn_results)
        #print("CNN SCORE "+str(cnn_results))
  
        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    if (index + 1) * batch_size > x_train.shape[0]:
        loss = model.train_on_batch(x=x_train[index * batch_size::],
                                            y=[p[index * batch_size::], expected_autoencoder_output[index * batch_size::]])
        index = 0
    else:
        loss = model.train_on_batch(x=x_train[index * batch_size:(index + 1) * batch_size],
                                            y=[p[index * batch_size:(index + 1) * batch_size],
                                            expected_autoencoder_output[index * batch_size:(index + 1) * batch_size]])
        index += 1
    ite += 1

y_pred_iter,_ =  model.predict(x_test)
y_pred_iter= y_pred_iter.argmax(1)
cnn_results=metrics.fowlkes_mallows_score(y_test,  y_pred_iter) 
print("CNN SCORE "+str(cnn_results))
cnn_score= np.append(cnn_score, cnn_results)

np.savetxt("cnn_scores.csv", cnn_score, fmt='%5s',delimiter=",")