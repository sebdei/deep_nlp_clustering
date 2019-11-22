from keras.models import Model, load_model
from keras.optimizers import SGD
import pandas as pd
import numpy as np
import tensorflow as tf

import clustering_utils
import preprocess
import text_provider

NUM_CLUSTERS = 5
AMOUNT_SEQUENCES = 2225

df = text_provider.provide_bbc_sequence_list()
embedding_matrix, padded_sequences = preprocess.preprocess_word_embedding(df.text)

# prevents memory issues on GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

autoencoder = load_model('models/autoencoder_trained.h5')

LAST_ENCODER_LAYER_KEY = "last_encoder_layer"
encoder_output = autoencoder.get_layer(LAST_ENCODER_LAYER_KEY).output
encoder = Model(inputs=autoencoder.inputs, outputs=encoder_output)

latent_features = encoder.predict(padded_sequences)
init_cluster_centers = clustering_utils.get_init_kmeans_cluster_centers(NUM_CLUSTERS, latent_features)

clustering_layer = clustering_utils.ClusteringLayer(NUM_CLUSTERS, weights=[init_cluster_centers], name='clustering')(encoder.output)
encoder_cluster_model = Model(inputs=encoder.input, outputs=clustering_layer)
encoder_cluster_model.compile(optimizer=SGD(0.01, 0.9), loss='kld')  # Kullback-leibner divergence loss

similarity_scores = encoder_cluster_model.predict(padded_sequences, verbose=0)
cluster_assignments = clustering_utils.get_cluster_assignments(similarity_scores)

clusterings_result = pd.DataFrame({'clustering_init': cluster_assignments})


# do Soft Assignment Hardening

max_iterations = 2500
update_interval = 120
batch_size = 16  # wrt to AMOUNT_SEQUENCES!!
index_array = np.arange(len(df.index))

losses = []
batch_index = 0
for i in range(int(max_iterations)):
    print("Iteration: %1d / %1d" % (i, max_iterations))
    similarity_scores = encoder_cluster_model.predict(padded_sequences, verbose=1)
    if i % update_interval == 0:
        target_distribution = clustering_utils.get_target_distribution(similarity_scores)
    if i % 60:
        clusterings_result['clustering_'+str(i)] = clustering_utils.get_cluster_assignments(similarity_scores)
        clusterings_result.to_csv('results/clustering_result.csv')
    idx = index_array[batch_index * batch_size: min((batch_index+1) * batch_size, padded_sequences.shape[0])]
    loss = encoder_cluster_model.train_on_batch(x=padded_sequences[idx], y=target_distribution[idx])
    losses.append(loss)
    batch_index = batch_index + 1 if (batch_index + 1) * batch_size <= padded_sequences.shape[0] else 0
