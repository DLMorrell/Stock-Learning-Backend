import trainer
import pandas as pd
from tensorflow.compat.v1.keras.models import Sequential, save_model, model_from_json
from tensorflow.compat.v1.keras.layers import Activation, Dense, Dropout, CuDNNLSTM, BatchNormalization
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
import numpy as np

from Tools.get_alpaca_market_data import market_data

# data params
window_len = 30#7
test_size = 0.2
zero_base = True
target_col = 'Close'

#Path to model
model_name = 'SPY-model-dropout-0.2-neurons-500x2-epochs-10-loss-mae-batch-16-win-10.json'
model_path = f'saved_models/{model_name}.json'
model_weight = f'saved_models/{model_name}.h5'

def load_data():
    print(f'Loading data...')
    df = pd.read_csv(f'SPY_All_Data.csv')
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Vol']
    hist = df.set_index('Time')
    hist.index = pd.to_datetime(hist.index)

    print(f'Load complete')
    return hist

def run(model):
    md = market_data()
    while True:
        md.waitForMarketToOpen()



if __name__ == "__main__":

    #Init
    #hist = load_data()

    #train, test, X_train, X_test, y_train, y_test = trainer.prepare_data(
    #hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

    #save data
    #trainer.save_data_to_numpy(X_train, X_test, y_train, y_test)

    #load model
    # model = keras.models.load_model(model_path)
    # model.load_weights(model_weight)

    # preds = model.predict(X_test)

    # for index, pred in enumerate(preds):
    #     print(f'{index} : {pred}, {y_test[index]}')