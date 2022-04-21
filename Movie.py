# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:30:40 2022

@author: admin
"""


import numpy as np 
import pandas as pd

movies = pd.read_csv('D:\movie-recommender-system-tmdb-dataset-main/tmdb_5000_movies.csv')
credits = pd.read_csv('D:\movie-recommender-system-tmdb-dataset-main/tmdb_5000_credits.csv') 

##Merge two files- 23 colums as its is joined using title so its common
movies = movies.merge(credits,on='title')

#removing all numeric and odd columns

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

#Null values
movies.isnull().sum()
 
# movie_id    0
# title       0
# overview    3
# genres      0
# keywords    0
# cast        0
# crew        0
# dtype: int64
movies.dropna(inplace=True)


movies.duplicated().sum()
#Out[15]: 0

##Our data is in Dict list. we have to transform to List
#1 convert string of list to list
import ast
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

##Actors
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L 

movies['cast'] = movies['cast'].apply(convert3)

##Top 3 actoes
#movies['cast'] = movies['cast'].apply(lambda x:x[0:3])

#fetch only director for crew
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 
movies['crew'] = movies['crew'].apply(fetch_director)

#STore list of words in overview

movies['overview'] = movies['overview'].apply(lambda x:x.split())
#we have to remove space from all as it treats as two diff tags for those.
#apply transform
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

#movies['overview'] = movies['overview'].apply(lambda x:[i.replace(" ","")for i in x])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
#drop col as tag is aggregation of all
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new['tags']=new['tags'].apply(lambda x:x.lower())

## as action and actions are same, we have to make them one
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new['tags']=new['tags'].apply(stem)
#vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
# as it returns sparse matrix, convert it to array
vector = cv.fit_transform(new['tags']).toarray()

cv.get_feature_names()

#euclidean dist is not a good measure for high dimensional data. Use cosine dist as dist in inversly proportional to similaritu
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

#take dist with index of movie and sort it. enumerate adds index num to list and then reverse it with sort on dist
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)        
        
recommend('Avatar')
import pickle

pickle.dump(new.to_dict(),open('movies_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
