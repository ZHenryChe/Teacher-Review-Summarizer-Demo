# Teacher Review Summarizer (Incomplete)

This project aims to summarize teacher reviews by identifying keywords in each review and identifying strengths/weaknesses of each teacher.
It uses a sentiment analysis model (finetuned distilBERT) to identify the tone of each comment, and a keyword analysis model (finetuned Flan_T5_base) to identify the keywords in each review.
Created for the AI Accelerator Program by AI Launch Lab, in the Fall 2024 cohort.

Repository template: [https://github.com/streamlit/blank-app-template]

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

   or depending on your needs

   ```
   $ streamlit run webpages\demo.py
   ```

   ```
   $ streamlit run webpages\from_txt_file.py
   ```

   ```
   $ streamlit run webpages\from_url.py
   ```
