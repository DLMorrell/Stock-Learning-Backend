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

sns.set_palette('Set2')
target_col = 'Close'
np.random.seed(42)

# data params
window_len = 30#7
test_size = 0.2
zero_base = True

# model params
lstm_neurons = 500
epochs = 10
batch_size = 16
loss = 'mae'
dropout = 0.20
optimizer = 'adam'

filepath = f"model-time-{time.time()}.h5"
model_name = f"SPY-model-dropout-{dropout}-neurons-{lstm_neurons}x2-epochs-{epochs}-loss-{loss}-batch-{batch_size}-win-{window_len}"

save_training_data = False
load_training_data = False
process_data = True

#endpoint = 'https://min-api.cryptocompare.com/data/histoday'
#res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=2000')
#hist = pd.DataFrame(json.loads(res.content)['Data'])
#hist = hist.set_index('time')
#hist.index = pd.to_datetime(hist.index, unit='s')

if process_data:
    print(f'Loading data...')
    df = pd.read_csv(f'SPY_all_data.csv')
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Vol']
    hist = df.set_index('Time')
    hist.index = pd.to_datetime(hist.index)

    print(f'Load complete')



def train_test_split(df, test_size=0.1):
    print(f'train_test_split...')
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
    print(f'extract_window_data...')
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

def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    print(f'prepare_data...')
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

def build_lstm_model(input_data, output_size, neurons=20, activ_func='linear',
                     dropout=0.25, loss='mae', optimizer='adam'):
    print(f'build_lstm_model...')
    model = Sequential()
    model.add(CuDNNLSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2]), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(CuDNNLSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

def save_data_to_numpy(X_train, X_test, y_train, y_test):
    print(f'Saving data as npy...')
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    print(f'Saved')

def load_data_from_numpy(file_location):
    print(f'Loading from {file_location}')
    X_train = np.load(f'{file_location}/X_train.npy')
    X_test= np.load(f'{file_location}/X_train.npy')
    y_train= np.load(f'{file_location}/X_train.npy')
    y_test= np.load(f'{file_location}/X_train.npy')

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = None, None, None, None

if process_data:
    train, test, X_train, X_test, y_train, y_test = prepare_data(
    hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

    if save_training_data:
        save_data_to_numpy(X_train, X_test, y_train, y_test)


# checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')

if load_training_data:
    X_train, X_test, y_train, y_test =load_data_from_numpy('data')


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)

history = model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True, callbacks=[tensorboard_callback])

targets = test[target_col][window_len:]
preds = model.predict(X_test).squeeze()

# 0.045261384613638447 with mae
print(f"mae: {mean_absolute_error(preds, y_test)}")

preds = test[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)

for i in range(len(targets)):
        print(f"Pred: {preds[i]}, Target: {targets[i]}")


n_points = 30

actual_returns = targets.pct_change()[1:]
predicted_returns = preds.pct_change()[1:]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

# actual correlation
corr = np.corrcoef(actual_returns, predicted_returns)[0][1]
ax1.scatter(actual_returns, predicted_returns, color='k', marker='o', alpha=0.5, s=100)
ax1.set_title('r = {:.2f}'.format(corr), fontsize=18)

# shifted correlation
shifted_actual = actual_returns[:-1]
shifted_predicted = predicted_returns.shift(-1).dropna()
corr = np.corrcoef(shifted_actual, shifted_predicted)[0][1]
ax2.scatter(shifted_actual, shifted_predicted, color='k', marker='o', alpha=0.5, s=100)
ax2.set_title('r = {:.2f}'.format(corr), fontsize=18);

#Saving model
model_json = model.to_json()
with open(model_name +".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_name + ".h5")
print("Saved model to disk")