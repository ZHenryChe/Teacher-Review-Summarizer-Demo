import streamlit as st
import time
from transformers import pipeline
from samples.txt_file_maker import scraping_baby_txt
import os
import nltk
import pandas as pd

nltk.download('punkt_tab')

#Model for sentiment analysis
sentiment_model = pipeline(model="ZephyrUtopia/ratemyprofessors-reviews-sentiment-analysis-10000-samples")

#Model for keyword extraction
keywordextract = pipeline("text2text-generation", model="ZephyrUtopia/keyword-summarizer-10000-v1", max_new_tokens=512)

#text display
st.write("""
        # RateMyProfessors Comment Summarizer.
         """)

url = st.text_input("Please input a valid RateMyProfessors URL.", value=None)

while url == None:
    time.sleep(1)

#get the reviews from the url
revs = scraping_baby_txt([url])

#dictionary to visualize final results
diction = {"Positive": [], 'Negative': []}

with st.spinner('Extracting keywords...'):

    for rev in revs:

        if rev:
            # Get the model prediction
            result = sentiment_model([rev])[0]  # Access the first result
            label = result['label']
            score = result['score'] * 100

            # Get rid of uncertain results
            if score <= 70:
                continue
    
            # Keyword extraction for the entire comment (you can also split it into smaller chunks if needed)
            keywords = keywordextract(f"Please find the keywords in this prompt: {rev}")

            # Cleaning keywords

            keywords = keywords[0]['generated_text']

            while True:
                start_pos = keywords.find("'")

                # Break loop if value not found
                if start_pos == -1:
                    break

                end_pos = keywords.find("'", start_pos+1)

                # Adding found word to dictionary
                if label == 'LABEL_0':
                    diction['Negative'].append(keywords[start_pos:end_pos+1])
                else:
                    diction['Positive'].append(keywords[start_pos:end_pos+1])

                # Truncating for next keyword
                keywords = keywords[end_pos+1:]

# this part to visualize results could be better graphically

#table to visualize results, orient+transpose to solve values not equal length problem
df = pd.DataFrame.from_dict(data = diction, orient = 'index').transpose()

# Display keywords
st.table(df)
