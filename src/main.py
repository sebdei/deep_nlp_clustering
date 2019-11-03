import csv
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from keras.utils import plot_model

from preprocess import preprocess_word_embedding


def define_lstm_autoencoder_layers(embedding_matrix, vocab_size, feature_dimension_size, max_sequence_length):
    result = Sequential()
    # encoder layers
    result.add(Embedding(vocab_size, feature_dimension_size, input_length=max_sequence_length, mask_zero=True, trainable=False, weights=[embedding_matrix]))
    result.add(LSTM(8, input_shape=(max_sequence_length, feature_dimension_size)))
    result.add(RepeatVector(max_sequence_length))
    # decoder layers
    result.add(LSTM(8, return_sequences=True))
    result.add(TimeDistributed(Dense(feature_dimension_size)))
    result.compile(optimizer='adam', loss='mse')
    result.summary()
    return result


with open('data/Reviews.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    dates = []
    sentence_list = [row[9] for row in readCSV]

sentence_list = sentence_list[:200]

embedding_matrix, padded_sequences = preprocess_word_embedding(sentence_list)

vocab_size = len(embedding_matrix)
feature_dimension_size = len(embedding_matrix[0])
max_sequence_length = len(padded_sequences[0])

lstm_autoencoder = define_lstm_autoencoder_layers(embedding_matrix, vocab_size, feature_dimension_size, max_sequence_length)

expected_output = np.array([[embedding_matrix[word_index] for word_index in encoded_sequence] for encoded_sequence in padded_sequences])
history = lstm_autoencoder.fit(padded_sequences, expected_output, epochs=10, verbose=1)

lstm_encoder = Model(inputs=lstm_autoencoder.inputs, outputs=lstm_autoencoder.layers[1].output)
# plot_model(lstm_encoder, show_shapes=True, to_file='lstm_encoder.png')

test = lstm_encoder.predict(padded_sequences)
print(test)
