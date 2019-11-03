import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from keras.utils import plot_model

from preprocess import preprocess_word_embedding

MAX_SEQUENCE_LENGTH = 6


def define_lstm_autoencoder_layers(embedding_matrix, vocab_size, feature_dimension_size):
    result = Sequential()

    # encoder layers
    result.add(Embedding(vocab_size, feature_dimension_size, input_length=MAX_SEQUENCE_LENGTH, mask_zero=True, trainable=False, weights=[embedding_matrix]))
    result.add(LSTM(2, activation='relu', input_shape=(MAX_SEQUENCE_LENGTH, feature_dimension_size)))
    result.add(RepeatVector(MAX_SEQUENCE_LENGTH))

    # decoder layers
    result.add(LSTM(2, activation='relu', return_sequences=True))
    result.add(TimeDistributed(Dense(feature_dimension_size)))

    result.compile(optimizer='adam', loss='mse')
    result.summary()

    return result


sentence_list = [
    'Well done!',
    'Good work',
    'Great effort',
    'nice work. your a good guy',
    'Excellent!',
    'Weak',
    'Poor effort!',
    'not good',
    'poor work',
    'Could have done better.'
    ]

embedding_matrix, padded_sequences, vocab_size, feature_dimension_size = preprocess_word_embedding(sentence_list)
lstm_autoencoder = define_lstm_autoencoder_layers(embedding_matrix, vocab_size, feature_dimension_size)

expected_output = np.array([[embedding_matrix[word_index] for word_index in encoded_sequence] for encoded_sequence in padded_sequences])
history = lstm_autoencoder.fit(padded_sequences, expected_output, epochs=50, verbose=1)

lstm_encoder = Model(inputs=lstm_autoencoder.inputs, outputs=lstm_autoencoder.layers[1].output)
plot_model(lstm_encoder, show_shapes=True, to_file='lstm_encoder.png')

test = lstm_encoder.predict(padded_sequences)
print(test)
