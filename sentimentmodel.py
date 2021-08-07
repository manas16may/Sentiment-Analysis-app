# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 23:49:57 2021

@author: manas
"""

import streamlit as st 
#import numpy as np 
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import nltk
import pickle
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import zipfile
zf = zipfile.ZipFile('IMDB Dataset.zip') 
data=pd.read_csv(zf.open('IMDB Dataset.csv'))
data=data[:5000]
d={"sentiment":{"positive":1,"negative":0}}
data=data.replace(d)
import re
#Removing the html strips
def strip_html(text):
         return re.sub('(<br\s*/><br\s*/>)|(\-)|(\/)', '', text)
    #Removing the square brackets
def remove_between_square_brackets(text):
            return re.sub('[.;:!\'?,\"()\[\]]', '', text)
    #Removing the noisy text
def denoise_text(text):
            text = strip_html(text)
            text = remove_between_square_brackets(text)
            return text
    #Apply function on review column
data['review']=data['review'].apply(denoise_text)
def remove_special_characters(text, remove_digits=True):
            pattern=r'[^a-zA-z0-9\s]'
            text=re.sub(pattern,'',text)
            return text
    #Apply function on review column
data['review']=data['review'].apply(remove_special_characters)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english'))
def remove(text):
            word_tokens = word_tokenize(text) 
            filtered_sentence = [w for w in word_tokens if not w in stop_words]   
            filtered_sentence = []  
            for w in word_tokens: 
                if w not in stop_words: 
                    filtered_sentence.append(w) 
                    filter=' '.join(filtered_sentence)
            return filter
data['review']=data['review'].apply(remove)
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
def lemma(text):
             word_tokens = word_tokenize(text)
             lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_tokens])
             return lemmatized_output
data['review']=data['review'].apply(lemma)
X_train=data.review[:3000]
X_test=data.review[3000:]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
stop_words = set(stopwords.words('english'))
vectorizer1 = TfidfVectorizer(stop_words=stop_words,max_features=2000)
cv_train =vectorizer1.fit_transform(X_train)
cv_test=vectorizer1.transform(X_test)
pickle_out1 = open("corpus.pkl","wb")
pickle.dump(vectorizer1, pickle_out1)
pickle_out1.close()
Y_train=data.sentiment[:3000]
Y_test=data.sentiment[3000:]
#st.write('Shape of dataset:', X.shape)
#st.write('number of classes:', len(np.unique(y)))
clf = LogisticRegression()
clf.fit(cv_train, Y_train)
y_pred = clf.predict(cv_test)
pickle_out = open("classifier.pkl","wb")
pickle.dump(clf, pickle_out)
pickle_out.close()
