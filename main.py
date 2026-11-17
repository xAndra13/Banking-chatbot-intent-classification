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