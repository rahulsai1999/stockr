import re
import datetime
import json
import pandas as pd
import tweepy
from flask import Flask
from pandas_datareader import data
from textblob import TextBlob
from tweepy import OAuthHandler

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


if __name__ == '__main__':
    app.run(debug=True)
