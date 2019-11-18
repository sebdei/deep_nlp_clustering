import os
import re
from nltk.tokenize import word_tokenize 
import model_provider
import pandas as pd
from preprocess import removeStopWords, createFastTextMatrix


#os.chdir("/Volumes/Files/Onedrive/Masters/Study Materials/Third Semester/Seminar-Recent Trends in Deep Learning")
#os.chdir("/Users/kevin/Downloads")


#Import dataset, data preprocessing
dataset = pd.read_csv("Reviews.csv")
dataset = dataset[['Score', 'Text']] 
dataset['Text'] = dataset['Text'].apply(lambda x : re.sub(r'[^\w\s]',"",re.sub("[!.#$%^&*()]","", x)).lower())
dataset['Text'] = dataset['Text'].apply(lambda x : removeStopWords(x))
dataset.to_csv("clean.csv") #save the intermediate dataframe
dataset = pd.read_csv("clean.csv")

#Get max words
dataset['Text'].map(len).max() #1975 words


matrix =  pd.DataFrame(columns=['Rating', 'CleanedText', 'SentenceMatrix'])
for i in range(len(dataset)):
    print("index") 
    print(i)
    print("array")
    matrix = matrix.append({'Rating': dataset.iloc[i,0], 'CleanedText': dataset.iloc[i,1],'WordMatrix':createFastTextMatrix(dataset.iloc[i,1])}, ignore_index=True)

matrix.to_pickle("Word_Matrices.pkl")