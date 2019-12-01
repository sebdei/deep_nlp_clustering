import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Embedding, LSTM, RepeatVector, TimeDistributed
from keras.models import Sequential

import preprocess
import text_provider

LAST_ENCODER_LAYER_KEY = "last_encoder_layer"
MODEL_BASE_NAME = "/autoencoder_trained_"
MODEL_PATH = "./models"


def plot_history(history, name=""):
    plt.plot(history.history["accuracy"])
    plt.title("Model accuracy " + name)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.show()
    plt.savefig("visualization/history_" + name + ".png")


def define_lstm_autoencoder_layers(embedding_matrix, vocab_size, feature_dimension_size, max_sequence_length, latent_feature_dimensions=32):
    autoencoder = Sequential(name="LSTM-Autoencoder")

    # encoder layers
    autoencoder.add(Embedding(vocab_size, feature_dimension_size, input_length=max_sequence_length, mask_zero=True, trainable=False, weights=[embedding_matrix]))
    autoencoder.add(LSTM(latent_feature_dimensions*2, activation="relu", return_sequences=True, input_shape=(max_sequence_length, feature_dimension_size)))
    autoencoder.add(LSTM(latent_feature_dimensions, return_sequences=False, name=LAST_ENCODER_LAYER_KEY))

    autoencoder.add(RepeatVector(max_sequence_length))  # Repeatvector for seq2seq lstm

    # decoder layers
    autoencoder.add(LSTM(latent_feature_dimensions, activation="relu", return_sequences=True))
    autoencoder.add(LSTM(latent_feature_dimensions*2, return_sequences=True))
    autoencoder.add(TimeDistributed(Dense(feature_dimension_size)))

    autoencoder.compile(optimizer="adam", loss="mse",  metrics=["accuracy"])
    autoencoder.summary()

    return autoencoder


def pretrain_lstm_autoencoder(latent_feature_dimensions=32):
    x_train, x_text, y_train, y_text = text_provider.provide_bbc_sequence_list()
    embedding_matrix, padded_sequences = preprocess.preprocess_word_embedding_fasttext(x_train)

    vocab_size = len(embedding_matrix)
    feature_dimension_size = len(embedding_matrix[0])
    max_sequence_length = len(padded_sequences[0])

    autoencoder = define_lstm_autoencoder_layers(
        embedding_matrix=embedding_matrix,
        vocab_size=vocab_size,
        feature_dimension_size=feature_dimension_size,
        max_sequence_length=max_sequence_length,
        latent_feature_dimensions=latent_feature_dimensions
    )

    expected_autoencoder_output = np.array([[embedding_matrix[word_index] for word_index in encoded_sequence] for encoded_sequence in padded_sequences])
    history = autoencoder.fit(padded_sequences, expected_autoencoder_output, epochs=40, verbose=2)
    autoencoder.save(MODEL_PATH + MODEL_BASE_NAME + str(latent_feature_dimensions*2) + "-" + str(latent_feature_dimensions) + ".h5")
    np.save(MODEL_PATH + MODEL_BASE_NAME + str(latent_feature_dimensions*2) + "-" + str(latent_feature_dimensions), history)
    plot_history(history, name=str(latent_feature_dimensions*2) + "-" + str(latent_feature_dimensions))

