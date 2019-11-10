import csv
import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import plot_model

from preprocess import preprocess_word_embedding
from clustering import ClusteringLayer, get_init_cluster_center
from lstm_autoencoder import define_lstm_autoencoder_layers
from utils import plot_features

with open('data/Reviews.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    sentence_list = [row[9] for row in readCSV]

sentence_list = sentence_list[:500]
embedding_matrix, padded_sequences = preprocess_word_embedding(sentence_list)

vocab_size = len(embedding_matrix)
feature_dimension_size = len(embedding_matrix[0])
max_sequence_length = len(padded_sequences[0])

autoencoder, encoder = define_lstm_autoencoder_layers(embedding_matrix, vocab_size, feature_dimension_size, max_sequence_length)
plot_model(autoencoder, show_shapes=True, to_file='visualisation/lstm_autoencoder.png')
plot_model(encoder, show_shapes=True, to_file='visualisation/lstm_encoder.png')

expected_output = np.array([[embedding_matrix[word_index] for word_index in encoded_sequence] for encoded_sequence in padded_sequences])
history = autoencoder.fit(padded_sequences, expected_output, epochs=20, verbose=1)


def get_target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def get_cluster_assignments(similarities):
    return [similarity_sample.argmax() for similarity_sample in similarities]


n_clusters = 2
latent_features = encoder.predict(padded_sequences)
init_cluster_centers = get_init_cluster_center(n_clusters, latent_features)

clustering_layer = ClusteringLayer(n_clusters, weights=[init_cluster_centers], name='clustering')(encoder.output)
cluster_model = Model(inputs=encoder.input, outputs=clustering_layer)
cluster_model.compile(optimizer=SGD(0.01, 0.9), loss='kld')  # Kullback-leibner divergence loss

similarities = cluster_model.predict(padded_sequences, verbose=0)
cluster_count = similarities.argmax(1)
cluster_assignments = get_cluster_assignments(similarities)

scales = {
    "min_x": min([latent_feature[0] for latent_feature in latent_features]) - 0.2,
    "max_x": max([latent_feature[0] for latent_feature in latent_features]) + 0.2,
    "min_y": min([latent_feature[1] for latent_feature in latent_features]) - 0.2,
    "max_y": max([latent_feature[1] for latent_feature in latent_features]) + 0.2
}
plot_features(features=latent_features, colors=cluster_assignments, cluster_assignments=cluster_assignments, scales=scales, i='init')

maxiter = 1100
update_interval = 120
batch_size = 126  # wrt sample_size!
index_array = np.arange(500)  # wrt sample_size!
losses = []
custer_assignment_counts = []

index = 0

for ite in range(int(maxiter)):
    print(ite)
    similarities = cluster_model.predict(padded_sequences, verbose=1)
    if ite % update_interval == 0:
        target_distribution = get_target_distribution(similarities)  # update the auxiliary target distribution p
        # evaluate the clustering performance
        # if y is not None:
        #     acc = np.round(metrics.acc(y, y_pred), 5)
        latent_features = encoder.predict(padded_sequences)
        cluster_assignments = get_cluster_assignments(similarities)
        plot_features(features=latent_features, colors=cluster_assignments, cluster_assignments=cluster_assignments, scales=scales, i=ite)
    idx = index_array[index * batch_size: min((index+1) * batch_size, padded_sequences.shape[0])]
    loss = cluster_model.train_on_batch(x=padded_sequences[idx], y=target_distribution[idx])
    losses.append(loss)
    index = index + 1 if (index + 1) * batch_size <= padded_sequences.shape[0] else 0
