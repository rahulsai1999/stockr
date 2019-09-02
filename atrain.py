# IMPORTING IMPORTANT LIBRARIES
import sys
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import preprocessing
import math


def train(ticker):
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="
    nextP = "&outputsize=full&apikey=T69K620H31T06293&datatype=csv"
    r = requests.get(url+ticker+nextP)
    path = 'temp/data/'+ticker+'.csv'
    pathm = 'temp/models/'+ticker+'.h5'

    with open(path, 'wb') as f:
        f.write(r.content)

    # FOR REPRODUCIBILITY
    np.random.seed(7)

    # IMPORTING DATASET
    dataset = pd.read_csv(path, usecols=[1, 2, 3, 4])
    dataset = dataset.reindex(index=dataset.index[::-1])

    # CREATING OWN INDEX FOR FLEXIBILITY
    obs = np.arange(1, len(dataset) + 1, 1)

    # TAKING DIFFERENT INDICATORS FOR PREDICTION
    OHLC_avg = dataset.mean(axis=1)
    HLC_avg = dataset[['high', 'low', 'close']].mean(axis=1)
    close_val = dataset[['close']]

    # PREPARATION OF TIME SERIES DATASE
    OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg), 1))  # 1664
    scaler = MinMaxScaler(feature_range=(0, 1))
    OHLC_avg = scaler.fit_transform(OHLC_avg)

    # TRAIN-TEST SPLIT
    train_OHLC = int(len(OHLC_avg) * 0.75)
    test_OHLC = len(OHLC_avg) - train_OHLC
    train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,
                                     :], OHLC_avg[train_OHLC:len(OHLC_avg), :]

    # TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
    trainX, trainY = preprocessing.new_dataset(train_OHLC, 1)
    testX, testY = preprocessing.new_dataset(test_OHLC, 1)

    # RESHAPING TRAIN AND TEST DATA
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    step_size = 1

    # LSTM MODEL
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, step_size), return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(1))
    model.add(Activation('linear'))

    # MODEL COMPILING AND TRAINING
    # Try SGD, adam, adagrad and compare!!!
    model.compile(loss='mean_squared_error', optimizer='adagrad')
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
    model.save(pathm)

    return ticker+ " Trained"

# ticker=sys.argv[1]
# train(ticker)
