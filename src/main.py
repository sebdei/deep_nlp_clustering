import numpy as np
from keras.models import Model
from keras.optimizers import SGD

import clustering
import lstm_autoencoder
import preprocess
import text_provider
import utils


NUM_CLUSTERS = 5
AMOUNT_SEQUENCES = 500

# sequence_list = text_provider.provide_sequence_list(amount=AMOUNT_SEQUENCES)
sequence_list = text_provider.provide_bbc_sequence_list()
sequence_list = sequence_list[0:500]
embedding_matrix, padded_sequences = preprocess.preprocess_word_embedding(sequence_list)
del sequence_list

vocab_size = len(embedding_matrix)
feature_dimension_size = len(embedding_matrix[0])
max_sequence_length = len(padded_sequences[0])

autoencoder, encoder = lstm_autoencoder.define_lstm_autoencoder_layers(embedding_matrix, vocab_size, feature_dimension_size, max_sequence_length)

expected_autoencoder_output = np.array([[embedding_matrix[word_index] for word_index in encoded_sequence] for encoded_sequence in padded_sequences])
history = autoencoder.fit(padded_sequences, expected_autoencoder_output, epochs=5, verbose=1)


# Autoencoder was trained - start clustering

latent_features = encoder.predict(padded_sequences)
init_cluster_centers = clustering.get_init_kmeans_cluster_centers(NUM_CLUSTERS, latent_features)

clustering_layer = clustering.ClusteringLayer(NUM_CLUSTERS, weights=[init_cluster_centers], name='clustering')(encoder.output)
encoder_cluster_model = Model(inputs=encoder.input, outputs=clustering_layer)
encoder_cluster_model.compile(optimizer=SGD(0.01, 0.9), loss='kld')  # Kullback-leibner divergence loss

similarity_scores = encoder_cluster_model.predict(padded_sequences, verbose=0)
cluster_assignments = clustering.get_cluster_assignments(similarity_scores)


# OPTIONAL: Plotting if latent features dimensions = 2
# plot_scales = utils.get_plot_scales(latent_features)
# utils.plot_2d_features(features=latent_features, cluster_assignments=cluster_assignments, scales=plot_scales, i='init')


# do Soft Assignment Hardening

max_iterations = 120
update_interval = 12
batch_size = 126  # wrt to AMOUNT_SEQUENCES!!
index_array = np.arange(AMOUNT_SEQUENCES)

losses = []
batch_index = 0
for i in range(int(max_iterations)):
    print("Iteration: %1d / %1d" % (i, max_iterations))
    print("predicting clusters")
    similarity_scores = encoder_cluster_model.predict(padded_sequences, verbose=1)
    if i % update_interval == 0:
        target_distribution = clustering.get_target_distribution(similarity_scores)
        # evaluate the clustering performance
        # if y is not None:
        #     acc = np.round(metrics.acc(y, y_pred), 5)
        latent_features = encoder.predict(padded_sequences)
        cluster_assignments = clustering.get_cluster_assignments(similarity_scores)
        # utils.plot_2d_features(features=latent_features, cluster_assignments=cluster_assignments, scales=plot_scales, i=i)
    idx = index_array[batch_index * batch_size: min((batch_index+1) * batch_size, padded_sequences.shape[0])]
    print("Train on batch")
    loss = encoder_cluster_model.train_on_batch(x=padded_sequences[idx], y=target_distribution[idx], verbose=1)
    losses.append(loss)
    batch_index = batch_index + 1 if (batch_index + 1) * batch_size <= padded_sequences.shape[0] else 0
