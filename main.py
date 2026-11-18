# Necesarry imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import count vectoriser
from sklearn.feature_extraction.text import CountVectorizer

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
print("No. of intents in train data: ", category_train.nunique())
print("No. of messages for each intent in train data: ", category_train.value_counts())
# check out some messages with the same intent
print(df_train[df_train['category'] == 'contactless_not_working']['text'])

# inspect test data
print("Head of test dataframe\n", df_test.head(), "\n")
print("No. of test entries: ", len(df_test))
inputs_test = df_test['text']
category_test = df_test['category']
print("No. of intents in test data: ", category_test.nunique())
print("No. of messages for each intent in test data: ", category_test.value_counts())
# see some messages with the same intent
print(df_test[df_test['category'] == 'contactless_not_working']['text'])

# count vectorising
cvect = CountVectorizer()
# fit and transform train data. STEPS:
# 1. it takes each word uniquely and uses it as 'key' for an internal dictionary he creates
# 2. each of those words, in the dictionary, is assigned the number of a column from a table it created like this:
"""
                                        account  card  contactless  deactivate  ...<the rest of the words>
I lost my card                            0       1            0           0   
Contactless doesn't work                  0       0            1           0   
How do i deactivate my own account        1       0            0           1
"""
# based on this table, the dictionary would look something like: {'account':0, 'card':1, 'contactless':2, etc.}

# transform the train data
x_train = cvect.fit_transform(inputs_train)
print(cvect.get_feature_names_out())
print(x_train) # toarray() is discoureged, as it consumes memory just to "shape" the data into a normal matrix; for even bigger data sets, it could cause a crash
print(x_train.shape)

# transform the test data
x_test = cvect.transform(inputs_test)
print(x_test)
print(x_test.shape)