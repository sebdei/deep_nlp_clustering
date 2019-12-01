from keras.models import Model, load_model
from keras.optimizers import SGD
import pandas as pd
import numpy as np
import tensorflow as tf

import clustering_utils
import preprocess
import text_provider
import pretrain_lstm_autoencoder

df = text_provider.provide_bbc_sequence_list()
embedding_matrix, padded_sequences = preprocess.preprocess_word_embedding_fasttext(df.text)

# prevents memory issues on GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

autoencoder = load_model(pretrain_lstm_autoencoder.MODEL_PATH)
encoder_output = autoencoder.get_layer(pretrain_lstm_autoencoder.LAST_ENCODER_LAYER_KEY).output
encoder = Model(inputs=autoencoder.inputs, outputs=encoder_output)

latent_features = encoder.predict(padded_sequences)

NUM_CLUSTERS = 5
init_cluster_centers = clustering_utils.get_init_kmeans_cluster_centers(NUM_CLUSTERS, latent_features)

clustering_layer = clustering_utils.ClusteringLayer(NUM_CLUSTERS, weights=[init_cluster_centers], name="clustering")(encoder.output)
encoder_cluster_model = Model(inputs=encoder.input, outputs=clustering_layer)
encoder_cluster_model.compile(optimizer='adam', loss="kld")  # Kullback-leibner divergence loss

similarity_scores = encoder_cluster_model.predict(padded_sequences, verbose=0)
cluster_assignments = clustering_utils.get_cluster_assignments(similarity_scores)


#  do Soft Assignment Hardening

batch_size = 16  # TODO: test with batch_size = 32 ?
max_iterations = 2801
update_interval = 140  # wrt to sequence_length e.g. 2225 / batch_size ?
index_array = np.arange(len(df.index))

losses = []
batch_index = 0

clusterings_result = pd.DataFrame()
for i in range(int(max_iterations)):
    print("Iteration: %1d / %1d" % (i, max_iterations))
    if i % update_interval == 0:
        similarity_scores = encoder_cluster_model.predict(padded_sequences)
        target_distribution = clustering_utils.get_target_distribution(similarity_scores)
        clusterings_result[str(i)] = clustering_utils.get_cluster_assignments(similarity_scores)
        clusterings_result.to_csv("results/clustering_result.csv")
    idx = index_array[batch_index * batch_size: min((batch_index+1) * batch_size, padded_sequences.shape[0])]
    loss = encoder_cluster_model.train_on_batch(x=padded_sequences[idx], y=target_distribution[idx])
    losses.append(loss)
    batch_index = batch_index + 1 if (batch_index + 1) * batch_size <= padded_sequences.shape[0] else 0

from sklearn.metrics import homogeneity_score