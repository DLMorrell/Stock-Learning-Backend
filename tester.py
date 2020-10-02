import json
import requests
from tensorflow.compat.v1.keras.models import Sequential, save_model, model_from_json
from tensorflow.compat.v1.keras.layers import Activation, Dense, Dropout, CuDNNLSTM, BatchNormalization
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import datetime, time
import tensorflow as tf

import re

endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=2000')
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')

window_len = 50#7
test_size = 0.1
zero_base = True

target_col = 'close'
loss = 'mae'
optimizer = 'adam'

list_of_model_names = ['model-dropout-0.15-neurons-50x2-epochs-5000-loss-mae-batch-128-win-4',
'model-dropout-0.15-neurons-50x2-epochs-5000-loss-mae-batch-128-win-25',
'model-dropout-0.15-neurons-500x2-epochs-5000-loss-mae-batch-256-win-50']

def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with respect to first entry. """
    return df / df.iloc[0] - 1

def normalise_min_max(df):
    """ Normalise dataframe column-wise min/max. """
    return (df - df.min()) / (df.max() - df.min())
def extract_window_data(df, window_len=10, zero_base=True):
    """ Convert dataframe to overlapping sequences/windows of len `window_data`.
    
        :param window_len: Size of window
        :param zero_base: If True, the data in each window is normalised to reflect changes
            with respect to the first entry in the window (which is then always 0)
    """
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

def load_model(model_name):
    json_file = open(f"{model_name}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(f"{model_name}.h5")
    model.compile(loss=loss, optimizer=optimizer) #added metics accuracy
    return model

def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    """ Prepare data for LSTM. """
    # train test split
    train_data, test_data = train_test_split(df, test_size=test_size)
    
    # extract window data
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    
    # extract targets
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test

if __name__ == "__main__":

    model_loss = {}
    model_pred = {}

    count = 0

    for model_name in list_of_model_names:
        count = count+1

        print(f'Testing #{count}...')
        window_len = int(re.findall(r'\d+', model_name)[-1])

        train, test, X_train, X_test, y_train, y_test = prepare_data(
        hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

        model = load_model(f'savedModels/{model_name}')

        preds = model.predict(X_test).squeeze()

        mae = mean_absolute_error(preds, y_test)

        model_loss[model_name] = mae

        for index in range(len(y_test)):
            print(f'pred: {preds[index]}, lable: {y_test[index]}')

    for item in model_loss.items():
        print(f"{item}")