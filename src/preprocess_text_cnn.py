import os
import re
from nltk.tokenize import word_tokenize 
import model_provider
import pandas as pd
import numpy as np
from preprocess import removeStopWords, createFastTextMatrix
from collections import Counter
import matplotlib.pyplot as plt


#os.chdir("/Volumes/Files/Onedrive/Masters/Study Materials/Third Semester/Seminar-Recent Trends in Deep Learning")
#os.chdir("/Users/kevin/Downloads")
#os.chdir("C:\\Users\\k_lim002\\Desktop\\Seminar")


#Import dataset, data preprocessing
dataset = pd.read_csv("Reviews.csv")
dataset = dataset[['reviews.rating', 'reviews.text']] 
dataset['reviews.text'] = dataset['reviews.text'].apply(lambda x : re.sub(r'[^\w\s\n]',"",re.sub("[!.#$%^&*()]","", str(x))).lower())
dataset['reviews.text'] = dataset['reviews.text'].apply(lambda x : removeStopWords(x))
dataset.to_csv("clean.csv") #save the intermediate dataframe
dataset = pd.read_csv("clean.csv") 

#Get max words
dataset['reviews.text'].map(len).max() #883 words


#Get length-- get length less than or equal to 365--
for i in range(len(dataset['reviews.text'])):
    dataset['length'][i] = len(dataset['reviews.text'][i])

length_counts = Counter(dataset['length'])
df = pd.DataFrame.from_dict(length_counts, orient='index')
df = df.reset_index()
df = df.sort_values(0, ascending=False)
df['CumSum'] =  df[0].cumsum()

plt.plot(df['CumSum'])
plt.show()


model= provide_fasttext_model()

matrix =  pd.DataFrame(columns=['Rating', 'CleanedText', 'SentenceMatrix'])
for i in range(len(dataset)):
    print("index") 
    print(i)
    matrix = matrix.append({'Rating': dataset.iloc[i,1], 'CleanedText': dataset.iloc[i,2],'WordMatrix':createFastTextMatrix(dataset.iloc[i,2])}, ignore_index=True)

matrix.to_pickle("Word_Matrices_small.pkl")

#Preprocessing of data for PCA and LSA
matrix = np.array([])

for i in range(len(dataset)):
    print(i)
    matrix = np.append(matrix, createFastTextArray(dataset.iloc[i,2]))


