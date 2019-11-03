import numpy as np
from keras.preprocessing.sequence import pad_sequences as keras_pad_sequenecs
from keras.preprocessing.text import Tokenizer

from model_provider import provide_glove_model
from model_provider import provide_fasttext_model


def build_embedding_matrix(word_index_dict, vocab_size, feature_dimension_size, model):
    result = np.zeros((vocab_size, feature_dimension_size))
    for word, i in word_index_dict.items():
        embedding_vector = model.get(word)
        if embedding_vector is not None:
            result[i] = embedding_vector

    return result


def preprocess_word_embedding(sentence_list):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence_list)

    word_index_dict = tokenizer.word_index
    vocab_size = len(word_index_dict) + 1  # +1 for zero padding ???

    glove_model = provide_glove_model()
    feature_dimension_size = 200
    embedding_matrix = build_embedding_matrix(word_index_dict, vocab_size, feature_dimension_size, glove_model)

    encoded_sequences = tokenizer.texts_to_sequences(sentence_list)
    padded_sequences = keras_pad_sequenecs(encoded_sequences, padding='post')

    return (embedding_matrix, padded_sequences)

def createFastTextMatrix(sentence):
    model = provide_fasttext_model()
    value = eval(sentence)
    print(len(value))
    embedding_matrix = np.zeros((23,300))
    for index in range(len(value)):
        embedding_matrix[index] = model.wv.get_vector(value[index])
    return embedding_matrix
    
