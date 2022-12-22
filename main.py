from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from better_profanity import profanity
# from sarcasm_predictor import predict
from sarcasm_new import predict_sarcasm
#from profanity_check import predict, predict_prob

# pf = ProfanityFilter()

# pf.censor("That's bullshit!")
# # "That's ********!"
data = pd.read_csv("labeled_data.csv")
#print(data.head())

data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
#print(data.head())

data = data[["tweet", "labels"]]
#print(data.head())

import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)
#print(data.head())

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

def hate_speech_detection():
    import streamlit as st
    st.title("Profanity Scanner App")
    user = st.text_area("Say your heart out...")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        st.header("Prediction using our basic ML Model:")
        a = clf.predict(data)
        st.subheader(a)
        st.header("Prediction using Better Model:")
        ans = profanity.contains_profanity(sample)
        sarcasm = predict_sarcasm(sample)
        # sarcasm = predict(sample)
        # print(sarcasm)
        if ans or sarcasm:
            st.subheader("Hate/Offensive/Sarcasm")
            profanity.load_censor_words()
            text_1 = sample
            censored_text = profanity.censor(text_1)
            st.subheader("After Censoring current text:")
            st.text(censored_text)
        
        else:
            st.subheader("No Hate or Offensive or Sarcasm Language")
            
hate_speech_detection()