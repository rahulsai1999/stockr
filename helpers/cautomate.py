import requests

stocks = ['AAPL', 'MSFT', 'FB', 'ADSK', 'AMZN']

res = []

for stock in stocks:
    url = "http://139.59.83.182:8000/stock/"+stock
    r = requests.get(url)
    print(r.content)
    res.append(r.content)

print(res)
