import csv
import re
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import StratifiedShuffleSplit


def provide_sequence_list(amount=-1):
    with open('data/Reviews.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        sentence_list = [row[9] for row in readCSV]

    return sentence_list[:amount]


def provide_bbc_sequence_list():
    data = load_files("./data/bbc/", encoding="utf-8", decode_error="replace")
    text = np.array([re.sub(r'[^\w\s\n]',"",re.sub("[!.#$%^&*()]","", str(x))).lower().replace('\n','').replace('\\','') for x in data['data']]).reshape(len(data['data']),1)
    label = data['target']
    stratSplit = StratifiedShuffleSplit(n_splits=5,test_size=0.4, random_state=42)
    for train_index, test_index in stratSplit.split(text, label):
        X_train, X_test = text[train_index], text[test_index]
        y_train, y_test = label[train_index], label[test_index]
    
    return X_train, X_test, y_train, y_test 
