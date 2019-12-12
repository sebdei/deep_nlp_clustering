import numpy as np
from keras.preprocessing.sequence import pad_sequences as keras_pad_sequenecs
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedShuffleSplit

import model_provider


def build_embedding_matrix(word_index_dict, vocab_size, feature_dimension_size, model):
    result = np.zeros((vocab_size, feature_dimension_size))
    for word, i in word_index_dict.items():
        embedding_vector = model.get(word)
        if embedding_vector is not None:
            result[i] = embedding_vector
    return result


def preprocess_word_embedding(sequence_list):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequence_list)

    word_index_dict = tokenizer.word_index
    vocab_size = len(word_index_dict) + 1  # +1 for zero padding ???

    glove_model = model_provider.provide_glove_model()
    feature_dimension_size = 200
    embedding_matrix = build_embedding_matrix(word_index_dict, vocab_size, feature_dimension_size, glove_model)

    encoded_sequences = tokenizer.texts_to_sequences(sequence_list)
    padded_sequences = keras_pad_sequenecs(encoded_sequences, padding='post')

    return (embedding_matrix, padded_sequences)


def build_embedding_matrix_fasttext(word_index_dict, vocab_size, feature_dimension_size, model):
    result = np.zeros((vocab_size, feature_dimension_size))
    for word, i in word_index_dict.items():
        try:
            embedding_vector = model.word_vec(word)
            if embedding_vector is not None:
                result[i] = embedding_vector
        except KeyError:
            pass  # embeddings with [0,0,...,0] are beeing ignored in LSTM layer due to masking
    return result


def preprocess_word_embedding_fasttext(sequence_list, label):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequence_list)
    word_index_dict = tokenizer.word_index
    vocab_size = len(word_index_dict) + 1

    model = model_provider.provide_fasttext_model()
    feature_dimension_size = 300
    embedding_matrix = build_embedding_matrix_fasttext(word_index_dict, vocab_size, feature_dimension_size, model)

    encoded_sequences = tokenizer.texts_to_sequences(sequence_list)
    padded_sequences = keras_pad_sequenecs(encoded_sequences, padding='post')

    stratSplit = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    for train_index, test_index in stratSplit.split(padded_sequences, label):
        x_train, x_test = padded_sequences[train_index], padded_sequences[test_index]
        y_train, y_test = label[train_index], label[test_index]

    return (embedding_matrix, x_train, x_test, y_train, y_test)
