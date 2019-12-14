import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Embedding, LSTM, RepeatVector, TimeDistributed
from keras.models import Sequential
import tensorflow as tf

import preprocess
import text_provider

LAST_ENCODER_LAYER_KEY = "last_encoder_layer"
MODEL_BASE_NAME = "/autoencoder_trained_"
MODEL_PATH = "./models"


def define_lstm_autoencoder_layers(embedding_matrix, vocab_size, feature_dimension_size, max_sequence_length, latent_feature_dimensions=32, loss="mse"):
    autoencoder = Sequential(name="LSTM-Autoencoder")

    # encoder layers
    autoencoder.add(Embedding(vocab_size, feature_dimension_size, input_length=max_sequence_length, mask_zero=True, trainable=False, weights=[embedding_matrix]))
    autoencoder.add(LSTM(latent_feature_dimensions*2, activation="relu", return_sequences=True, input_shape=(max_sequence_length, feature_dimension_size)))
    autoencoder.add(LSTM(latent_feature_dimensions, activation="relu", return_sequences=False, name=LAST_ENCODER_LAYER_KEY))

    autoencoder.add(RepeatVector(max_sequence_length))  # Repeatvector for seq2seq lstm

    # decoder layers
    autoencoder.add(LSTM(latent_feature_dimensions, activation="relu", return_sequences=True))
    autoencoder.add(LSTM(latent_feature_dimensions*2, activation="relu", return_sequences=True))
    autoencoder.add(TimeDistributed(Dense(feature_dimension_size)))

    autoencoder.compile(optimizer="adam", loss=loss,  metrics=["mse", "mae", "cosine_proximity"])
    autoencoder.summary()

    return autoencoder


def pretrain_lstm_autoencoder(latent_feature_dimensions=32, loss="mse"):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

    text, label = text_provider.provide_bbc_sequence_list()
    embedding_matrix, x_train, x_test, y_train, y_test = preprocess.preprocess_word_embedding_fasttext(text, label)

    vocab_size = len(embedding_matrix)
    feature_dimension_size = len(embedding_matrix[0])
    max_sequence_length = len(x_train[0])

    autoencoder = define_lstm_autoencoder_layers(
        embedding_matrix=embedding_matrix,
        vocab_size=vocab_size,
        feature_dimension_size=feature_dimension_size,
        max_sequence_length=max_sequence_length,
        latent_feature_dimensions=latent_feature_dimensions,
        loss=loss
    )

    expected_autoencoder_output = np.array([[embedding_matrix[word_index] for word_index in encoded_sequence] for encoded_sequence in x_train])
    history = autoencoder.fit(x_train, expected_autoencoder_output, epochs=75, verbose=1)
    autoencoder.save(MODEL_PATH + MODEL_BASE_NAME + str(latent_feature_dimensions*2) + "-" + str(latent_feature_dimensions) + "_" + loss + ".h5")
    np.save(MODEL_PATH + MODEL_BASE_NAME + str(latent_feature_dimensions*2) + "-" + str(latent_feature_dimensions), history)
