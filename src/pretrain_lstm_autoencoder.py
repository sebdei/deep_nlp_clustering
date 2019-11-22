import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Embedding, LSTM, RepeatVector, TimeDistributed
from keras.models import Sequential

import preprocess
import text_provider

LAST_ENCODER_LAYER_KEY = "last_encoder_layer"
MODEL_PATH = "./models/autoencoder_trained.h5"


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()
    plt.savefig('visualization/history.png')


def define_lstm_autoencoder_layers(embedding_matrix, vocab_size, feature_dimension_size, max_sequence_length):
    autoencoder = Sequential(name="LSTM-Autoencoder")

    # encoder layers
    autoencoder.add(Embedding(vocab_size, feature_dimension_size, input_length=max_sequence_length, mask_zero=True, trainable=False, weights=[embedding_matrix]))
    autoencoder.add(LSTM(64, activation="relu", return_sequences=True, input_shape=(max_sequence_length, feature_dimension_size)))
    autoencoder.add(LSTM(32, return_sequences=False, name=LAST_ENCODER_LAYER_KEY))

    autoencoder.add(RepeatVector(max_sequence_length))  # Repeatvector for seq2seq lstm

    # decoder layers
    autoencoder.add(LSTM(32, activation="relu", return_sequences=True))
    autoencoder.add(LSTM(64, return_sequences=True))
    autoencoder.add(TimeDistributed(Dense(feature_dimension_size)))

    autoencoder.compile(optimizer='adam', loss='mean_absolute_error',  metrics=['accuracy'])
    autoencoder.summary()

    return autoencoder


def pretrain_lstm_autoencoder():
    df = text_provider.provide_bbc_sequence_list()
    embedding_matrix, padded_sequences = preprocess.preprocess_word_embedding(df.text)

    vocab_size = len(embedding_matrix)
    feature_dimension_size = len(embedding_matrix[0])
    max_sequence_length = len(padded_sequences[0])

    autoencoder = define_lstm_autoencoder_layers(
        embedding_matrix=embedding_matrix,
        vocab_size=vocab_size,
        feature_dimension_size=feature_dimension_size,
        max_sequence_length=max_sequence_length,
        )

    expected_autoencoder_output = np.array([[embedding_matrix[word_index] for word_index in encoded_sequence] for encoded_sequence in padded_sequences])
    history = autoencoder.fit(padded_sequences, expected_autoencoder_output, epochs=50, verbose=1)
    autoencoder.save('models/autoencoder_trained')
    plot_history(history)
