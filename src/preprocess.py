import numpy as np
from keras.preprocessing.sequence import pad_sequences as keras_pad_sequenecs
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize 
import nltk
from nltk.corpus import stopwords

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


model = provide_fasttext_model()
def createFastTextMatrix(sentence):
    global model 
    #model = model_provider.provide_fasttext_model()
    value = eval(sentence)
    print(len(value))
    embedding_matrix = np.zeros((900, 300))
    for index in range(len(value)):
        embedding_matrix[index] = model.wv.get_vector(value[index])
    return embedding_matrix


def build_embedding_matrix_fasttext(word_index_dict, vocab_size, feature_dimension_size, model):
    # Too slow because the model is too large.
    result = np.zeros((vocab_size, feature_dimension_size))
    for word, i in word_index_dict.items():
        embedding_vector = model.wv.get_vector(word)
        if embedding_vector is not None:
            result[i] = embedding_vector

    return result


def preprocess_word_embedding_fasttext(sequence_list):
    # Too slow because the model is too large.
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequence_list)

    word_index_dict = tokenizer.word_index
    vocab_size = len(word_index_dict) + 1

    model = model_provider.provide_fasttext_model()
    feature_dimension_size = 300
    embedding_matrix = build_embedding_matrix_fasttext(word_index_dict, vocab_size, feature_dimension_size, model)

    encoded_sequences = tokenizer.texts_to_sequences(sequence_list)
    padded_sequences = keras_pad_sequenecs(encoded_sequences, padding='post')

    return (embedding_matrix, padded_sequences)

def removeStopWords(text):
    en_stop = set(stopwords.words('english'))
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in en_stop] 
    return filtered_sentence