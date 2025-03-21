import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK datasets if not available
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()


def transform_text(sent):
    sent = sent.lower()
    text = nltk.word_tokenize(sent)
    res = [word for word in text if word.isalnum()]
    res = [word for word in res if word not in stopwords.words('english') and word not in string.punctuation]
    res = [ps.stem(word) for word in res]
    return " ".join(res)


# Streamlit UI
st.set_page_config(page_title='SMS SpamWire', page_icon='📩', layout='centered')

st.markdown(
    """
  <style>
      .stApp {background-color: #ADD8E6;}
      .title {text-align: center; font-size: 100px; font-weight: bold; color: #FF5733;}
      .result {font-size: 20px; padding: 10px; text-align: center; border-radius: 5px;}
      .not_spam {background-color: #d4edda; color: #155724;}
      .spam {background-color: #f8d7da; color: #721c24;}
  </style>
  """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">📩 SMS SpamWire</p>', unsafe_allow_html=True)

st.image("https://cdn-icons-png.flaticon.com/512/1039/1039286.png", width=100)

input_msg = st.text_area("Enter the SMS", height=100)

if st.button("Check Message"):
    if input_msg.strip():
        preprocessed_msg = transform_text(input_msg)
        vectorised_msg = tfidf.transform([preprocessed_msg])
        pred = model.predict(vectorised_msg)

        if pred[0] == 0:
            st.markdown('<p class="result not_spam">✅ Not a Spam Message</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result spam">🚨 Beware! It\'s a Spam</p>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message to analyze.")
