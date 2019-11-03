# from numpy import array
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed
# from keras.utils import plot_model

# # define input sequence
# sequence = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# # reshape input into [samples, timesteps, features]
# n_in = len(sequence)
# sequence = sequence.reshape((1, n_in, 1))
# print(sequence)

# # define model
# model = Sequential()
# model.add(LSTM(100, activation='relu', input_shape=(n_in, 1)))
# model.add(RepeatVector(n_in))
# model.add(LSTM(100, activation='relu', return_sequences=True))
# model.add(TimeDistributed(Dense(1)))
# model.compile(optimizer='adam', loss='mse')
# # fit model
# model.fit(sequence, sequence, epochs=300, verbose=0)

# # plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# # demonstrate recreation
# yhat = model.predict(sequence, verbose=0)
# print(yhat[0,:,0])

# Tokenizer 
from keras.preprocessing.text import Tokenizer
text = 'this is a test bla!'
tokenizer = Tokenizer(nb_words=5)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)


def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size, 200))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = glove.get(word)
    return weight_matrix


# lstm autoencoder recreate sequence
from numpy import array, zeroes
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
import matplotlib.pyplot as plt

from model_provider import provide_glove_model
from keras.layers import Embedding

# words = [
#     'this',
#     'is',
#     'a',
#     'test'
#     ]
# word_embeddings = [model.get(word) for word in words]

glove = provide_glove_model()

sentences = [
    ['this', 'is', 'a', 'test'],
    ['i', 'hate', 'you', 'and', 'test'],
]



word_embeddings = [[glove.get(word) for word in sentence] for sentence in sentences]
sequence = array(word_embeddings)
padded_input = pad_sequences(sequence, padding='post')
padded_inputs = pad_sequences(raw_inputs, padding='post')


# # define input sequence
# # reshape input into [samples, timesteps, features]
# n_in = len(sequence[0])
# print(n_in)

sequence = sequence.reshape((1, 2, 200))  # 2 sentences, 4 timesteps, 200 number of features


# ENCODER
model = Sequential()
embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
model.add(LSTM(50, activation='relu', input_shape=(4, 200)))  # 200 dimensions, 2 sentences
model.add(RepeatVector(200))  # 200 dimensions

# DECODER
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(4)))  # 2 sentences
model.compile(optimizer='adam', loss='mse')

history = model.fit(sequence, sequence, epochs=50, verbose=0)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print(model.summary())

# connect the encoder LSTM as the output layer
model = Model(inputs=model.inputs, outputs=model.layers[0].output)


plot_model(model, show_shapes=True, to_file='lstm_encoder.png')
# get the feature vector for the input sequence
yhat = model.predict(sequence)
# print(yhat.shape)
# print(yhat)
