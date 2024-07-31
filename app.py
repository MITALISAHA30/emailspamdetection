import streamlit as st
import pickle

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("Email Spam Classifier")
input_sms = st.text_area("Enter message")

if st.button('Predict'):
    if input_sms == "":
        st.error("Please enter message!")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]


        if result == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")


