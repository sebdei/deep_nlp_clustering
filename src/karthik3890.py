import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

sample = pd.read_csv("data/Reviews.csv")
sample = sample[0:500]


def partition(x):
    if x < 3:
        return 'negative'
    return 'positive'


actual_score = sample['Score']
positiveNegative = actual_score.map(partition)
sample['Score'] = positiveNegative

sample["Score"].value_counts()

sorted_data = sample.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
final = sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

final = final[final.HelpfulnessNumerator <= final.HelpfulnessDenominator]


i = 0
for sent in final['Text'].values:
    if (len(re.findall('<.*?>', sent))):
        print(i)
        print(sent)
        break
    i += 1


nltk.download('stopwords')
sno = nltk.stem.SnowballStemmer('english')  #initialising the snowball stemmer which is developed in recent years
stop = set(stopwords.words('english'))


def cleanhtml(sentence):  # function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext


def cleanpunc(sentence):  # function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    return cleaned


print(stop)
print('************************************')
print(sno.stem('tasty'))

i = 0
str1 = ' '
final_string = []
all_positive_words = []  # store words from +ve reviews here
all_negative_words = []  # store words from -ve reviews here.
s = ''

for sent in final['Text'].values:
    filtered_sentence = []
    #print(sent);
    sent = cleanhtml(sent)  # remove HTMl tags
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words) > 2)):
                if(cleaned_words.lower() not in stop):
                    s = (sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i] == 'positive':
                        all_positive_words.append(s)  # list of all words used to describe positive reviews
                    if(final['Score'].values)[i] == 'negative':
                        all_negative_words.append(s)  # list of all words used to describe negative reviews reviews
                else:
                    continue
            else:
                continue
    # print(filtered_sentence)
    str1 = b" ".join(filtered_sentence)  # final string of cleaned words
    # print("***********************************************************************")
    final_string.append(str1)
    i += 1


final['CleanedText'] = final_string  # adding a column of CleanedText which displays the data after pre-processing of the review 
final['CleanedText'] = final['CleanedText'].str.decode("utf-8")

count_vect = CountVectorizer()
bow = count_vect.fit_transform(final['CleanedText'].values)
bow.shape

model = KMeans(n_clusters=10, init='k-means++', n_jobs=-1, random_state=99)
model.fit(bow)
