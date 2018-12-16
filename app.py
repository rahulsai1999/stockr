from flask import Flask
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

@app.route("/<ticker>")
def hello(ticker):
    start_date = '2017-01-01'
    end_date = '2018-12-14'
    panel_data = data.DataReader(ticker, 'yahoo', start_date, end_date)

    return panel_data.to_json()

if __name__ == '__main__':
    app.run(debug=True)