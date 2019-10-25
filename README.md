# Stock Price Prediction of Apple Inc. Using Recurrent Neural Network

[![Build Status](https://travis-ci.com/rahulsai1999/stockr.svg?branch=master)](https://travis-ci.com/rahulsai1999/stockr)

OHLC Average Prediction of Apple Inc. Using LSTM Recurrent Neural Network

# Dataset:
The dataset is taken from yahoo finace's website in CSV format. The dataset consists of Open, High, Low and Closing Prices of Apple Inc. stocks.
# Price Indicator:
Stock traders mainly use three indicators for prediction: OHLC average (average of Open, High, Low and Closing Prices), HLC average (average of High, Low and Closing Prices) and Closing price, In this project, OHLC average has been used.
# Data Pre-processing:
After converting the dataset into OHLC average, it becomes one column data. This has been converted into two column time series data, 1st column consisting stock price of time t, and second column of time t+1. All values have been normalized between 0 and 1.
# Model: 
Two sequential LSTM layers have been stacked together and one dense layer is used to build the RNN model using Keras deep learning library. Since this is a regression task, 'linear' activation has been used in final layer.
# Version:
Python 3.7 and latest versions of all libraries including deep learning library Keras and Tensorflow.
# Training:
75% data is used for training. Adagrad (adaptive gradient algorithm) optimizer is used for faster convergence.
# Test:
Test accuracy metric is root mean square error (RMSE).

# Observation and Conclusion:
Since difference among OHLC average, HLC average and closing value is not significat, so only OHLC average is used to build the model and prediction. The training and testing RMSE are: 1.24 and 1.37 respectively which is pretty good to predict future values of stock.
Finally, this work can greatly help the quantitative traders to take decisions.

