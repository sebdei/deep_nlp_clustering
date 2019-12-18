import csv
import re
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
import preprocess
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')


def provide_amazon_sequence_list():
    reviews = pd.read_csv("/home/dobby/deep_nlp_clustering/src/data/Reviews.csv")
    reviews = reviews[reviews['reviews.rating'] > 0]
    reviews['reviews.text'] = reviews['reviews.text'].apply(lambda x: re.sub(r'[^\w\s\n]', "", re.sub("[!.#$%^&*()]", "", str(x))).lower())
    reviews['reviews.text'] = reviews['reviews.text'].apply(lambda x: preprocess.removeStopWords(x))
    label = reviews['reviews.rating']
    data = reviews['reviews.text']

    return data, label


def provide_bbc_sequence_list():
    data = load_files("./data/bbc/", encoding="utf-8", decode_error="replace")
    data['data'] = np.array([removeStopWords(x) for x in data['data']])
    text = np.array([re.sub(r'[^\w\s\n]', "", re.sub("[!.#$%^&*()]", "", str(x))).lower().replace('\n', '').replace('\\', '') for x in data['data']]).reshape(len(data['data']), 1)
    label = data['target']

    return text.flatten(), label


def provide_sequence_list(dataset='bbc'):
    if (dataset == 'bbc'):
        text, label = provide_bbc_sequence_list()
    elif (dataset == 'amazon_reviews'):
        text, label = provide_amazon_sequence_list()

    return text, label


def removeStopWords(text):
    en_stop = set(stopwords.words('english'))
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in en_stop]
    return filtered_sentence
