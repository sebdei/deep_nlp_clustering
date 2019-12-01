import os
import re
from nltk.tokenize import word_tokenize 
import model_provider
import pandas as pd
import numpy as np
from preprocess import removeStopWords, createFastTextMatrix
from collections import Counter
import matplotlib.pyplot as plt
from text_provider.py import provide_bbc_sequence_list
from model_provider import provide_fasttext_model


#os.chdir("/Volumes/Files/Onedrive/Masters/Study Materials/Third Semester/Seminar-Recent Trends in Deep Learning")
#os.chdir("/Users/kevin/Downloads")
#os.chdir("C:\\Users\\k_lim002\\Desktop\\Seminar")


#Import dataset, data preprocessing for amazon
dataset = pd.read_csv("Reviews.csv")
dataset = dataset[['reviews.rating', 'reviews.text']] 
dataset['reviews.text'] = dataset['reviews.text'].apply(lambda x : re.sub(r'[^\w\s\n]',"",re.sub("[!.#$%^&*()]","", str(x))).lower())
dataset['reviews.text'] = dataset['reviews.text'].apply(lambda x : removeStopWords(x))
dataset.to_csv("amazon.csv") #save the intermediate dataframe
dataset = pd.read_csv("amazon.csv") 

#Import dataset, data preprocessing for BBC
dataset= provide_bbc_sequence_list()
dataset['text'] = dataset['text'].apply(lambda x : re.sub(r'[^\w\s\n]',"",re.sub("[!.#$%^&*()]","", str(x))).lower().replace('\n','').replace('\\',''))
dataset['text'] = dataset['text'].apply(lambda x : removeStopWords(x))
dataset.to_csv("bbc.csv") #save the intermediate dataframe
dataset = pd.read_csv("bbc.csv") 


#Get length--
lengths= np.array([len(dataset['text'][i]) for i in range(len(dataset['text']))])
#BBC Max Length 23142

length_counts = Counter(dataset['length'])
df = pd.DataFrame.from_dict(length_counts, orient='index')
df = df.reset_index()
df = df.sort_values(0, ascending=False)
df['CumSum'] =  df[0].cumsum()

plt.plot(df['CumSum'])
plt.show()


model= provide_fasttext_model()

dimension=23142
matrix =  pd.DataFrame(columns=['Rating', 'CleanedText', 'SentenceMatrix'])
for i in range(len(dataset)):
    print("index") 
    print(i)
    matrix = matrix.append({'Rating': dataset.iloc[i,2], 'WordMatrix':createFastTextMatrix(dataset.iloc[i,1],dimension)}, ignore_index=True)

matrix.to_pickle("BBC_Word_Matrices_max.pkl")



