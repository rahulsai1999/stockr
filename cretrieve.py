import redis
r = redis.Redis(host='redis',port=6379)


def getval(ticker):
    la = r.get(ticker+"last")
    ne = r.get(ticker+"next")
    y = {'last': la, 'next': ne}
    return y
