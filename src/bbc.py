import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.metrics import homogeneity_score


# for reproducibility
random_state = 0

data = load_files("./data/bbc/", encoding="utf-8", decode_error="replace")
df = pd.DataFrame(list(zip(data['data'], data['target'])), columns=['text', 'label'])
df.head()

df['cleaned_text'] = df.text.str.replace('\n\n', '\n', regex=False)
df['cleaned_text'] = df.cleaned_text.str.replace('\n', ' ', regex=False)

vec = TfidfVectorizer(stop_words="english")
vec.fit(df.cleaned_text.values)
features = vec.transform(df.cleaned_text.values)

cls = MiniBatchKMeans(n_clusters=5, random_state=random_state)
cls.fit(features)

df['y_tf_idf'] = cls.predict(features)

homogeneity_score(df.label, cls.predict(features))

df.groupby('label')['y_tf_idf'].value_counts()
