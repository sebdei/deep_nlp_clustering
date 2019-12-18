from keras.models import Model, load_model
import pandas as pd
import numpy as np
import tensorflow as tf

import clustering_utils
import preprocess
import text_provider
import pretrain_lstm_autoencoder


def do_cluster_hardening(model_file_name, dataset="bbc"):
    text, label = text_provider.provide_sequence_list(dataset)
    embedding_matrix, x_train, x_test, y_train, y_test = preprocess.preprocess_word_embedding_fasttext(text, label)

    # prevents memory issues on GPU
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

    autoencoder = load_model(pretrain_lstm_autoencoder.MODEL_PATH + "/" + model_file_name)
    encoder_output = autoencoder.get_layer(pretrain_lstm_autoencoder.LAST_ENCODER_LAYER_KEY).output
    encoder = Model(inputs=autoencoder.inputs, outputs=encoder_output)

    NUM_CLUSTERS = len(np.unique(label))
    latent_features = encoder.predict(x_train)
    init_cluster_centers = clustering_utils.get_init_kmeans_cluster_centers(NUM_CLUSTERS, latent_features)

    clustering_layer = clustering_utils.ClusteringLayer(NUM_CLUSTERS, weights=[init_cluster_centers], name="clustering")(encoder.output)
    encoder_cluster_model = Model(inputs=encoder.input, outputs=clustering_layer)
    encoder_cluster_model.compile(optimizer='adam', loss="kld")  # Kullback-leibner divergence loss

    batch_size = 16  # TODO: test with batch_size = 32 ?
    max_iterations = 2801
    update_interval = 140  # wrt to train size e.g. 2225 / batch_size ?
    index_array = np.arange(len(x_train))
    batch_index = 0

    clusterings_result = pd.DataFrame()
    for i in range(int(max_iterations)):
        print("Iteration: %1d / %1d" % (i, max_iterations))
        if i % update_interval == 0:
            similarity_scores = encoder_cluster_model.predict(x_train)
            clusterings_result[str(i)] = clustering_utils.get_cluster_assignments(similarity_scores)
            target_distribution = clustering_utils.get_target_distribution(similarity_scores)
            clusterings_result.to_csv("cluster_results/" + model_file_name + ".csv")
        idx = index_array[batch_index * batch_size: min((batch_index+1) * batch_size, x_train.shape[0])]
        encoder_cluster_model.train_on_batch(x=x_train[idx], y=target_distribution[idx])
        batch_index = batch_index + 1 if (batch_index + 1) * batch_size <= x_train.shape[0] else 0

    encoder_cluster_model.save(pretrain_lstm_autoencoder.MODEL_PATH + "/finished_cluster_models/" + model_file_name)
