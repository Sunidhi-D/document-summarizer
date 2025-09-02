import streamlit as st
import wikipedia
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import evaluate
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

st.title("Document Summerizer")

# Input options
use_file = st.checkbox("Use uploaded text file instead of Wikipedia")
if use_file:
    uploaded_file = st.file_uploader("Upload text file (.txt)", type="txt")
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
else:
    query = st.text_input("Enter Wikipedia Topic")
    if query:
        try:
            text = wikipedia.page(query).content
            st.write(f"✅ Wikipedia article for '{query}' fetched successfully!")
        except Exception as e:
            st.error(f"Error fetching Wikipedia page: {e}")
            text = ""

# Summarization
if st.button("Summarize") and text:
    st.write(f"Original Text Length: {len(text)} characters")

    # Extractive Summary
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    cleaned_words = [w for w in words if w not in stop_words and w not in string.punctuation]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    scores = np.array(X.sum(axis=1)).ravel()
    ranked_sentence = [sent for _, sent in sorted(zip(scores, sentences), reverse=True)]
    extractive_summary = " ".join(ranked_sentence[:5])

    st.subheader("Extractive Summary")
    st.write(extractive_summary)

    # Abstractive Summary
    summarizer = pipeline(
        "summarization",
        model=r"C:\Users\Sunidhi\local_distilbart_model",
        tokenizer=r"C:\Users\Sunidhi\local_distilbart_model"
    )

    summary = []
    chunk_size = 1500
    for i in range(0, len(text), chunk_size):
        part_out = summarizer(text[i:i+chunk_size], max_length=50, min_length=20, do_sample=False)
        summary.append(part_out[0]['summary_text'])

    final_summary = " ".join(summary)

    st.subheader("Abstractive Summary")
    st.write(final_summary)

    # ROUGE score (optional, using extractive summary as reference)
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=[final_summary], references=[extractive_summary])
    st.subheader("ROUGE Scores (vs Extractive Summary)")
    st.write(scores)

    # Download
    st.download_button(
        label="Download Abstractive Summary",
        data=final_summary,
        file_name="abstractive_summary.txt",
        mime="text/plain"
    )
