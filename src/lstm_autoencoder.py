from keras.layers import Dense, Embedding, LSTM, RepeatVector, TimeDistributed
from keras.models import Model, Sequential

LAST_ENCODER_LAYER = "last_encoder_layer"


def define_lstm_autoencoder_layers(embedding_matrix, vocab_size, feature_dimension_size, max_sequence_length):
    lstm_autoencoder = Sequential(name="LSTM-Autoencoder")

    # encoder layers
    lstm_autoencoder.add(Embedding(vocab_size, feature_dimension_size, input_length=max_sequence_length, mask_zero=True, trainable=False, weights=[embedding_matrix]))
    lstm_autoencoder.add(LSTM(32, input_shape=(max_sequence_length, feature_dimension_size), return_sequences=True))
    lstm_autoencoder.add(LSTM(2, return_sequences=False, name=LAST_ENCODER_LAYER))
    lstm_autoencoder.add(RepeatVector(max_sequence_length))  # Repeatvector for seq2seq lstm

    # decoder layers
    lstm_autoencoder.add(LSTM(2, return_sequences=True))
    lstm_autoencoder.add(LSTM(32, return_sequences=True))
    lstm_autoencoder.add(TimeDistributed(Dense(feature_dimension_size)))
    lstm_autoencoder.compile(optimizer='adam', loss='mean_squared_error',  metrics=['accuracy'])
    lstm_autoencoder.summary()

    encoder_output = lstm_autoencoder.get_layer(LAST_ENCODER_LAYER).output
    lstm_encoder = Model(inputs=lstm_autoencoder.inputs, outputs=encoder_output)

    return lstm_autoencoder, lstm_encoder
