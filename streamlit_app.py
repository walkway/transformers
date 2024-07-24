import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Initialize NLP tools
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

# Load summarization pipeline
summarizer = pipeline("summarization")

# Function to summarize text
def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to extract keywords
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    keywords = list(set(keywords))  # Remove duplicates
    return keywords

# Streamlit app layout
st.title("Text Summarizer and Keyword Extractor")

st.write("""
## Enter Text
""")

user_input = st.text_area("Paste your text here", height=200)

if st.button("Summarize and Extract Keywords"):
    if user_input:
        summary = summarize_text(user_input)
        keywords = extract_keywords(user_input)
        
        st.write("### Summary")
        st.write(summary)
        
        st.write("### Keywords")
        st.write(", ".join(keywords))
    else:
        st.write("Please enter some text to summarize and extract keywords.")