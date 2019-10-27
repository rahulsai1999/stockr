# IMPORTING IMPORTANT LIBRARIES
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
import preprocessing
import math
import redis

np.random.seed(7)
r = redis.Redis(host='redis',port=6379)


def pred(ticker):

    path = 'temp/data/'+ticker+'.csv'
    pathm = 'temp/models/'+ticker+'.h5'

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

    model = load_model(pathm)

    # PREDICTION
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # DE-NORMALIZING FOR PLOTTING
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # TRAINING RMSE
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    # print('Train RMSE: %.2f' % (trainScore))

    # TEST RMSE
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    # print('Test RMSE: %.2f' % (testScore))

    # CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
    trainPredictPlot = np.empty_like(OHLC_avg)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

    # CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
    testPredictPlot = np.empty_like(OHLC_avg)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(step_size*2) +
                    1:len(OHLC_avg)-1, :] = testPredict

    # DE-NORMALIZING MAIN DATASET
    OHLC_avg = scaler.inverse_transform(OHLC_avg)

    # PREDICT FUTURE VALUES
    last_val = testPredict[-1]
    last_val_scaled = last_val/last_val
    next_val = model.predict(np.reshape(last_val_scaled, (1, 1, 1)))
    finobj = {'last': np.asscalar(
        last_val), 'next': np.asscalar(last_val*next_val)}
    r.set(ticker+"last", finobj['last'])
    r.set(ticker+'next', finobj['next'])
    return {'result': ticker+" Values Stored"}


# ticker = sys.argv[1]
# print(pred(ticker))
