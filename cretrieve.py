import redis
r = redis.Redis()


def getval(ticker):
    la = r.get(ticker+"last")
    ne = r.get(ticker+"next")
    y = {'last': la, 'next': ne}
    return y
