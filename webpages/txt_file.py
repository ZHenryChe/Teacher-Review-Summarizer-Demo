import streamlit as st
import wx
from transformers import pipeline
import nltk
import pandas as pd

nltk.download('punkt_tab')

#Model for sentiment analysis
sentiment_model = pipeline(model="ZephyrUtopia/ratemyprofessors-reviews-sentiment-analysis-10000-samples")

#Model for keyword extraction
keywordextract = pipeline("text2text-generation", model="ZephyrUtopia/keyword-summarizer-10000-v1", max_new_tokens=512)


#text display
st.write("""
        # Please upload a file to summarize.
         """)

#file selector
if st.button('Browse'):
    dialog = wx.DirDialog(None, 'Select a folder:', style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
    if dialog.ShowModal() == wx.ID_OK:
        folder_path = dialog.GetPath() # folder_path will contain the path of the folder you have selected as string

revs = open(folder_path, 'r')

revs_split = revs.read().split('\n\n')  #comments to review

#dictionary to visualize final results
diction = {"Positive": [], 'Negative': []}

for rev in revs_split:

    with st.spinner('Extracting keywords...'):

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
                if label == 0:
                    diction['Negative'].append(keywords[start_pos:end_pos+1])
                else:
                    diction['Positive'].append(keywords[start_pos:end_pos+1])

                # Truncating for next keyword
                keywords = keywords[end_pos+1:]


#table to visualize results
df = pd.DataFrame(data = diction)

# Display keywords
st.table(df)
