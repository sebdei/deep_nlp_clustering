import csv
import pandas as pd
from sklearn.datasets import load_files


def provide_sequence_list(amount=-1):
    with open('data/Reviews.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        sentence_list = [row[9] for row in readCSV]

    return sentence_list[:amount]


def provide_bbc_sequence_list():
    data = load_files("./data/bbc/", encoding="utf-8", decode_error="replace")
    return pd.DataFrame(list(zip(data['data'], data['target'])), columns=['text', 'label'])
