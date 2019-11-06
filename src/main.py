import csv
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from keras.utils import plot_model

from preprocess import preprocess_word_embedding
import matplotlib.pyplot as plt


def define_lstm_autoencoder_layers(embedding_matrix, vocab_size, feature_dimension_size, max_sequence_length):
    result = Sequential()
    # encoder layers
    result.add(Embedding(vocab_size, feature_dimension_size, input_length=max_sequence_length, mask_zero=True, trainable=False, weights=[embedding_matrix]))
    result.add(LSTM(32, input_shape=(max_sequence_length, feature_dimension_size), return_sequences=True))
    result.add(LSTM(3, return_sequences=False))
    result.add(RepeatVector(max_sequence_length))
    # decoder layers
    result.add(LSTM(3, return_sequences=True))
    result.add(LSTM(32, return_sequences=True))
    result.add(TimeDistributed(Dense(feature_dimension_size)))
    result.compile(optimizer='adam', loss='mean_squared_error',  metrics=['accuracy'])
    result.summary()
    return result


with open('data/Reviews.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    dates = []
    sentence_list = [row[9] for row in readCSV]

sentence_list = sentence_list[:600]

embedding_matrix, padded_sequences = preprocess_word_embedding(sentence_list)

vocab_size = len(embedding_matrix)
feature_dimension_size = len(embedding_matrix[0])
max_sequence_length = len(padded_sequences[0])

lstm_autoencoder = define_lstm_autoencoder_layers(embedding_matrix, vocab_size, feature_dimension_size, max_sequence_length)

expected_output = np.array([[embedding_matrix[word_index] for word_index in encoded_sequence] for encoded_sequence in padded_sequences])
history = lstm_autoencoder.fit(padded_sequences, expected_output, epochs=15, verbose=1)


plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('Model accuratcy')
plt.savefig('plots/model_performance.png')
plt.clf()

lstm_encoder = Model(inputs=lstm_autoencoder.inputs, outputs=lstm_autoencoder.layers[1].output)
# plot_model(lstm_encoder, show_shapes=True, to_file='lstm_encoder.png')

latent_features = lstm_encoder.predict(padded_sequences)

x = [datapoint[0] for datapoint in latent_features]
y = [datapoint[1] for datapoint in latent_features]
z = [datapoint[2] for datapoint in latent_features]

plt.clf()
plt.scatter(x, y, c=z, alpha=0.5)
plt.savefig('plots/plot.png')
plt.clf()