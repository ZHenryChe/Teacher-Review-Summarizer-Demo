import streamlit as st
from transformers import pipeline
from rake_nltk import Rake
import time
import nltk
nltk.download('punkt_tab')

sentiment_model = pipeline(model="ZephyrUtopia/ratemyprofessors-reviews-sentiment-analysis-10000-samples")

# User input for the comment
comment = st.text_input("Enter your comment:", value = None)

while comment == None:
    time.sleep(1)
    
if comment:
    # Get the model prediction
    result = sentiment_model([comment])[0]  # Access the first result
    label = result['label']
    score = result['score'] * 100

    # Define sentiment based on the label
    sentiment = "positive" if label == "LABEL_1" else "negative"  # "LABEL_1" corresponds to 1 (positive)
    
    # Display the result
    st.write(f"The comment is {sentiment} (certainty: {score:.2f})")





## Now, process the keywords from the comment using the keyword extractor
keywordextract = pipeline("text2text-generation", model="ZephyrUtopia/keyword-summarizer-10000-v1", max_new_tokens=512)

with st.spinner('Extracting keywords...'):
            # Keyword extraction for the entire comment (you can also split it into smaller chunks if needed)
            keywords = keywordextract(f"Please find the keywords in this prompt: {comment}")

            # Display keywords
            st.write("Extracted Keywords:", keywords[0]['generated_text'])
