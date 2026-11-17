# Necesarry imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data set that was used - Banking77
# ensure internet connection for data downloading
url = 'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/'
# read the train and test data sets
df_train = pd.read_csv(url + 'train.csv')
df_test = pd.read_csv(url + 'test.csv')

# inspect train data
print("Head of train dataframe\n", df_train.head(), "\n")
print("No. of train entries: ", len(df_train))
inputs_train = df_train['text']
category_train = df_train['category']
print("No. intents in train data: ", category_train.nunique())
print("No. of messages for each intent in train data:", category_train.value_counts())
# check out some messages with the same intent
print(df_train[df_train['category'] == 'contactless_not_working']['text'])