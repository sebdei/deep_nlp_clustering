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
    encoder_cluster_model = Model(inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])
    encoder_cluster_model.compile(optimizer='adam', loss=['kld', 'cosine_proximity'], loss_weights=[0.1, 0.9])

    expected_autoencoder_output = np.array([[embedding_matrix[word_index] for word_index in encoded_sequence] for encoded_sequence in x_train])

    # Hyperparams
    batch_size = 16
    max_iterations = 2223
    update_interval = 111

    index_array = np.arange(len(x_train))
    batch_index = 0

    clusterings_result_train = pd.DataFrame()
    clusterings_result_test = pd.DataFrame()

    for i in range(int(max_iterations)):
        print("Iteration: %1d / %1d" % (i, max_iterations))
        if i % update_interval == 0:
            similarity_scores_train, _ = encoder_cluster_model.predict(x_train)
            target_distribution = clustering_utils.get_target_distribution(similarity_scores_train)

            clusterings_result_train[str(i)] = clustering_utils.get_cluster_assignments(similarity_scores_train)
            clusterings_result_train.to_csv("cluster_results/" + model_file_name + "_train.csv")

            similarity_scores_test, _ = encoder_cluster_model.predict(x_test)
            clusterings_result_test[str(i)] = clustering_utils.get_cluster_assignments(similarity_scores_test)
            clusterings_result_test.to_csv("cluster_results/" + model_file_name + "_test.csv")

        idx = index_array[batch_index * batch_size: min((batch_index+1) * batch_size, x_train.shape[0])]
        encoder_cluster_model.train_on_batch(x=x_train[idx], y=[target_distribution[idx], expected_autoencoder_output[idx]])
        batch_index = batch_index + 1 if (batch_index + 1) * batch_size <= x_train.shape[0] else 0

    encoder_cluster_model.save(pretrain_lstm_autoencoder.MODEL_PATH + "/finished_cluster_models/" + model_file_name)
