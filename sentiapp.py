# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 23:42:49 2021

@author: manas
"""

import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#app=Flask(__name__)
#Swagger(app)
pickle_in = open("D:/sentimentanalyzer/classifier.pkl","rb")
clf=pickle.load(pickle_in)
cv = pickle.load(open('D:/sentimentanalyzer/corpus.pkl','rb'))
def welcome():
    return "Welcome All"
def cleaning(review):
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
    review=denoise_text(review)
    def remove_special_characters(text, remove_digits=True):
            pattern=r'[^a-zA-z0-9\s]'
            text=re.sub(pattern,'',text)
            return text
    #Apply function on review column
    review=remove_special_characters(review)
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
            filter1=' '.join(filtered_sentence)
            return filter1
    review=remove(review)
    from nltk.stem import WordNetLemmatizer 
    lemmatizer = WordNetLemmatizer()
    def lemma(text):
             word_tokens = word_tokenize(text)
             lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_tokens])
             return lemmatized_output
    review1=lemma(review)
    list1=[" "]
    list1[0]=review1
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    c=cv.transform(list1)
    return c
def predict(new_review):
        new_review = re.sub('[^a-zA-Z]', ' ', new_review)
        new_review = new_review.lower()
        new_review = new_review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
        new_review = ' '.join(new_review)
        new_corpus = [new_review]
        new_X_test = cv.transform(new_corpus).toarray()
        prediction=clf.predict(new_X_test)
        return prediction
def main():
   st.title("Sentiment Analysis")
   html_temp = """
    <div style="background-color:orange;padding:10px">
    <h2 style="color:white;text-align:center;">Sentiment Analysis ML App </h2>
    </div>
    """ 
   st.header('This app is created to predict if a review is positive or negative')
   st.markdown(html_temp,unsafe_allow_html=True)
   review = st.text_input("review","Type Here")
   result=""
   if st.button("Predict"):
      result=predict(review)
      if result==0:
          result="negative"
      if result==1:
          result="positive"
   st.success('The review is {}'.format(result))
if __name__=='__main__':
    main()