import csv
import re
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import StratifiedShuffleSplit
import preprocess


def provide_sequence_list(amount=-1):
    with open('data/Reviews.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        sentence_list = [row[9] for row in readCSV]

    return sentence_list[:amount]

def provide_bbc_sequence_list():
    data = load_files("./data/bbc/", encoding="utf-8", decode_error="replace")
    data['data'] = np.array([preprocess.removeStopWords(x) for x in data['data']])
    
    text = np.array([re.sub(r'[^\w\s\n]', "", re.sub("[!.#$%^&*()]", "", str(x))).lower().replace('\n', '').replace('\\', '') for x in data['data']]).reshape(len(data['data']), 1)
    
    label = data['target']

    return text.flatten(), label

def provide_amazon_sequence_list():
    reviews =pd.read_csv("/home/dobby/deep_nlp_clustering/src/data/Reviews.csv")
    reviews = reviews[reviews['reviews.rating']>0]
    reviews['reviews.text'] = reviews['reviews.text'].apply(lambda x : re.sub(r'[^\w\s\n]',"",re.sub("[!.#$%^&*()]","", str(x))).lower())
    reviews['reviews.text'] = reviews['reviews.text'].apply(lambda x : preprocess.removeStopWords(x))
    label = reviews['reviews.rating']
    data = reviews['reviews.text']

    return data, label