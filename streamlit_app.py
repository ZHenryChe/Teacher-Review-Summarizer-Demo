import streamlit as st
from streamlit.logger import get_logger

import requests
from bs4 import BeautifulSoup

import json

import pyarrow
from datasets import Dataset

from datasets import load_dataset

import pandas as pd

from transformers import AutoTokenizer

from transformers import DataCollatorWithPadding

from transformers import AutoModelForSequenceClassification

import numpy as np
from evaluate import load

from huggingface_hub import notebook_login

from transformers import TrainingArguments, Trainer

from transformers import pipeline

import nltk
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

import random




st.title("Datasets")
st.subheader('Training data set')
st.write('10500 range')

#training dataset
base_dataset = load_dataset("ZephyrUtopia/ratemyprofessors_reviews")
base_dataset_train = base_dataset['train'].shuffle().select(range(10500))

pd_dataset_train = pd.DataFrame(base_dataset_train)
pd_dataset_train.head()

#remove names from reviews (would need to iterate)
pd_dataset_train2 = pd_dataset_train

for row in range(len(pd_dataset_train)):

  name_lst= pd_dataset_train['name'][row].split()

  for name in name_lst:
    pd_dataset_train2.loc[row, "text"] = pd_dataset_train.loc[row, 'text'].replace(name, '')
    pd_dataset_train2.loc[row, "text"] = pd_dataset_train.loc[row, 'text'].replace('  ', ' ')

pd_dataset_train2.head()



st.write(pd_dataset_train2)


st.subheader('Testing data set')
st.write('1500 range')
#testing dataset
base_dataset_test = base_dataset['test'].shuffle().select(range(1500))
pd_dataset_test = pd.DataFrame(base_dataset_test)

for row in range(len(pd_dataset_test)):

  name_lst= pd_dataset_test['name'][row].split()

  for name in name_lst:
    pd_dataset_test.loc[row, "text"] = pd_dataset_test.loc[row, 'text'].replace(name, '')
    pd_dataset_test.loc[row, "text"] = pd_dataset_test.loc[row, 'text'].replace('  ', ' ')

POSITIVE_LST = ['4.0', '5.0']
NEGATIVE_LST = ['1.0', '2.0']

row = 0
while row < len(pd_dataset_test):

  if pd_dataset_test['rating'][row] in NEGATIVE_LST:
    pd_dataset_test.loc[row, "rating"] = 0

  elif pd_dataset_test['rating'][row] in POSITIVE_LST:
    pd_dataset_test.loc[row, "rating"] = 1

  else:
    pd_dataset_test = pd_dataset_test.drop(row)
    pd_dataset_test = pd_dataset_test.reset_index(drop=True)
    row -= 1

  row += 1

pd_dataset_test = pd_dataset_test.drop(columns= ['name', 'difficulty', 'date'])
pd_dataset_test = pd_dataset_test.rename(columns= {'rating': 'label'})

st.write(pd_dataset_test)
