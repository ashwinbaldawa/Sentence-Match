import string
import pandas as pd
import seaborn as sns
import nltk
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import time

# Calculate Start Time
start_time = time.time()

# Reading the input CSV file

color = sns.color_palette()
train_df = pd.read_csv(r"C:\Users\ashwin\PycharmProjects\similarity\Match Finder\sample\Train_new.csv",nrows= 1000)

# Setting the Threshold for Cosine Similarity
Match_Thr = 0.4

# Making a separate column just to make one column for matching
df_train = train_df[['question1']].copy()

# Convert to lower case
df_train['question1'] = list(map(lambda x: x.lower(), df_train['question1']))

# remove all the numerics
df_train['question1'] = df_train['question1'].replace(to_replace = '\d', value = "",regex= True)

# tokenize the data
df_train['question1'] = df_train['question1'].apply(word_tokenize)

# Stop words processing
stop = stopwords.words('english')
df_train['question1'] = df_train['question1'].apply(lambda x: [item for item in x if item not in stop])

# Remove Bad characters
bad_chars = [';', ':', '!', "*", "?","[", "]","(",")","{","}",",","/","^",".",","]
df_train['question1'] = df_train['question1'].apply(lambda x: [item for item in x if item not in bad_chars])

# Stemming the data
stemmer = SnowballStemmer("english")
df_train['question1'] = df_train['question1'].apply(lambda x: [stemmer.stem(y) for y in x])

# joining back the entire sentence
df_train['question1'] = df_train['question1'].str.join(" ")

# Apply TDIf and then Cosine Similarity
vectorizer = TfidfVectorizer()
trsfm = vectorizer.fit_transform(df_train['question1'])
trsfm_df =pd.DataFrame(trsfm)

# Apply cosine similarity
cos_sim = cosine_similarity(trsfm, trsfm)

# Below logic is to find the optimum clusters using Silhoutte technique

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# This is the minimum possible score
sil_score_max = -1

for n_clusters in range(2,30):
  model = KMeans(n_clusters = n_clusters, init='k-means++', max_iter=100, n_init=1)
  labels = model.fit_predict(trsfm)
  sil_score = silhouette_score(trsfm, labels)
  if sil_score > sil_score_max:
    sil_score_max = sil_score
    best_n_clusters = n_clusters

# flatten the array and get the value with match > 0.6 and less than 1,
# this i value is the threshold which can be given from the UI
kmeans = KMeans(n_clusters = best_n_clusters)
kmeans.fit(trsfm)
clusters = kmeans.labels_.tolist()

question_fin = train_df['question1'].tolist()
final_col = {'cluster': clusters, 'question1': question_fin}
Final_temp = pd.DataFrame(final_col,columns= ['cluster','question1'])
cos_df = pd.DataFrame(cos_sim)

# Creating a concatenated Dataframe with Similarity Values
for col in cos_df:
    for i, row_value in cos_df[col].iteritems():
        if row_value <= Match_Thr:
            cos_df[col][i] = 0

final_df = pd.concat([Final_temp,cos_df],axis=1)
#print(final_df.head())
# Extracting the matched questions
match = []
pd_value =[]
question = []
count = -1

for indr,rows in final_df.iloc[:,1:].head(n=500).iterrows():
     a = rows.to_numpy().nonzero()
     values = rows.iloc[a]
     match.append(list((values.index)[1:]))
     pd_value.append(list(values[1:]))
     count += 1
     question.append(len(values[1:]) * [count])

# import chain
from itertools import chain

# converting 2d list into 1d
# using chain.from_iterables

Question_list = list(chain.from_iterable(question))
Match_list    = list(chain.from_iterable(match))
Cosine_Val    = list(chain.from_iterable(pd_value))

Cosine_col = {'Question': Question_list, 'Matched Que': Match_list,'Cosine Val': Cosine_Val}
Cosine_df = pd.DataFrame(Cosine_col,columns= ['Question','Matched Que','Cosine Val'])
Cosine_df = Cosine_df[~(Cosine_df.astype(int)['Question'] == Cosine_df.astype(int)['Matched Que'])]

for i in match:
    a = match.index(i)
    print(a)
    i.remove(a)

Final_temp['Target'] = match
edges_temp = Final_temp.copy()

print(pd_value)

for j in pd_value:
    j.astype(int).remove(1)

print(pd_value)
edges_temp['Cosine'] = pd_value
edges = edges_temp[edges_temp.astype(str)['Cosine'] != '[]']

# Convert the Final dataframe to CSV
Final_temp.to_csv('Nodes.csv')
edges.to_csv('Edges.csv')
Cosine_df.to_csv('Cosine_df.csv')

#import pickle
#with open('model_pickle','wb') as f:
#    pickle.dump(model,f)

print("--- %s seconds ---" % (time.time() - start_time))