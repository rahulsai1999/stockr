import re
import datetime
import json
import pandas as pd
import tweepy
from flask import Flask
from pandas_datareader import data
from textblob import TextBlob
from tweepy import OAuthHandler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras import backend
import preprocessing
import numpy as np

app = Flask(__name__)

class TwitterClient(object): 
	''' 
	Generic Twitter Class for sentiment analysis. 
	'''
	def __init__(self): 
		''' 
		Class constructor or initialization method. 
		'''
		# keys and tokens from the Twitter Dev Console 
		consumer_key = 'AmiUfJdtgulb97IIerLZxYXbZ'
		consumer_secret = 'Ag79Lvhm8gFkZRWhzhWVYdEMXLo9SGqDMwvglzM68U0Zx82xEE'
		access_token = '1533856164-7612ddsggnXH7fzHm8KhHg1OaGfprtH8MHvPDeB'
		access_token_secret = 'G9ijoarXmA8EIXaWsBxfG0XbgDA2LeTdK3icuEinTNlJI'

		# attempt authentication 
		try: 
			# create OAuthHandler object 
			self.auth = OAuthHandler(consumer_key, consumer_secret) 
			# set access token and secret 
			self.auth.set_access_token(access_token, access_token_secret) 
			# create tweepy API object to fetch tweets 
			self.api = tweepy.API(self.auth) 
		except: 
			print("Error: Authentication Failed") 

	def clean_tweet(self, tweet): 
		''' 
		Utility function to clean tweet text by removing links, special characters 
		using simple regex statements. 
		'''
		return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z\t])|(\w+:\/\/\S+)", " ", tweet).split()) 

	def get_tweet_sentiment(self, tweet): 
		''' 
		Utility function to classify sentiment of passed tweet 
		using textblob's sentiment method 
		'''
		# create TextBlob object of passed tweet text 
		analysis = TextBlob(self.clean_tweet(tweet)) 
		# set sentiment 
		if analysis.sentiment.polarity > 0: 
			return 'positive'
		elif analysis.sentiment.polarity == 0: 
			return 'neutral'
		else: 
			return 'negative'

	def get_tweets(self, query, count = 10): 
		''' 
		Main function to fetch tweets and parse them. 
		'''
		# empty list to store parsed tweets 
		tweets = [] 

		try: 
			# call twitter api to fetch tweets 
			fetched_tweets = self.api.search(q = query, count = count) 

			# parsing tweets one by one 
			for tweet in fetched_tweets: 
				# empty dictionary to store required params of a tweet 
				parsed_tweet = {} 

				# saving text of tweet 
				parsed_tweet['text'] = tweet.text 
				# saving sentiment of tweet 
				parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) 

				# appending parsed tweet to tweets list 
				if tweet.retweet_count > 0: 
					# if tweet has retweets, ensure that it is appended only once 
					if parsed_tweet not in tweets: 
						tweets.append(parsed_tweet) 
				else: 
					tweets.append(parsed_tweet) 

			# return parsed tweets 
			return tweets 

		except tweepy.TweepError as e: 
			# print error (if any) 
			print("Error : " + str(e))

@app.route("/")
def hxx():
    return "/stock/MSFT for stock data and /nlp/query for NLP data"

@app.route("/stock/<ticker>")
def hello(ticker):
    start_date = '2018-12-01'
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    panel_data = data.DataReader(ticker, 'yahoo', start_date, end_date)

    return panel_data.to_json()

@app.route("/nlp/<qurry>")
def findeverything(qurry):
    # creating object of TwitterClient Class 
	api = TwitterClient() 
	# calling function to get tweets 
	tweets = api.get_tweets(query = qurry, count = 1000) 

	# picking positive tweets from tweets 
	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
    
	ptweper=len(ptweets)/len(tweets)
    
	ntweper=len(ntweets)/len(tweets)
    
	finobj={'p':ptweper,'n':ntweper,'ptwe':ptweets,'ntwe':ntweets}

	return json.dumps(finobj)

@app.route("/stockpr/apple")
def predictstcg():
    model = load_model('stockmodel.h5')
    dataset = pd.read_csv("AAPL.csv", usecols=[1,2,3,4])
    dataset = dataset.reindex(index = dataset.index[::-1])

    # CREATING OWN INDEX FOR FLEXIBILITY
    obs = np.arange(1, len(dataset) + 1, 1)

    # TAKING DIFFERENT INDICATORS FOR PREDICTION
    OHLC_avg = dataset.mean(axis = 1)
    HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
    close_val = dataset[['Close']]

    # PREPARATION OF TIME SERIES DATASE
    OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) # 1664
    scaler = MinMaxScaler(feature_range=(0, 1))
    OHLC_avg = scaler.fit_transform(OHLC_avg)

    # TRAIN-TEST SPLIT
    train_OHLC = int(len(OHLC_avg) * 0.75)
    test_OHLC = len(OHLC_avg) - train_OHLC
    train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

    # TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
    trainX, trainY = preprocessing.new_dataset(train_OHLC, 1)
    testX, testY = preprocessing.new_dataset(test_OHLC, 1)

    # RESHAPING TRAIN AND TEST DATA
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    step_size = 1

    # PREDICTION
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # DE-NORMALIZING FOR PLOTTING
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    last_val = testPredict[-1]
    last_val_scaled = last_val/last_val
    next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
    backend.clear_session()
    return str(np.asscalar(last_val-5))

if __name__ == '__main__':
    app.run(debug=True)
