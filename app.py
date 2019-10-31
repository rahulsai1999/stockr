from flask import Flask
from flask_cors import CORS
import simplejson as json
import atrain
import bpredict
import cretrieve
import nlp

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def fx():
    return "Hello"


@app.route('/stock/<ticker>')
def trainModel(ticker):
    x = atrain.train(ticker)
    y = bpredict.pred(ticker)
    return json.dumps(y)


@app.route('/stockpr/<ticker>')
def predictModel(ticker):
    y = cretrieve.getval(ticker)
    return json.dumps(y)


@app.route('/nlp/<qurry>')
def ex(qurry):
    y = nlp.findeverything(qurry)
    return json.dumps(y)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
