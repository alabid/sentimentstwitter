#! /usr/bin/env python
import tornado.ioloop
import tornado.web
import urllib
import tweepy
import os
from hidden import *

from naivebayesclassifier import NaiveBayesClassifier

# name of training set file
fname = 'trainingandtestdata/training.csv'

# train classifiers here first
nb = NaiveBayesClassifier(fname, grams=[1,2])
nb.setThresholds(neg=1.0, pos=20.0)
nb.setWeight(0.000000000005)
nb.trainClassifier()

# ment = MaximumEntropyClassifier(fname)
# ment.ge
# classifiers = [nb, ment]
classifiers = [nb]

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        query = self.get_argument("query", "").strip()
        cchosen = int(self.get_argument("classifier", 0))

        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        
        api = tweepy.API(auth)

        results = api.search(q=urllib.quote(query)) if len(query) > 0 else []

        tweets = []
        poscount = 0
        negcount = 0
        for result in results:
            cresult = classifiers[cchosen].classify(result.text)

            if cresult == 0: negcount += 1
            elif cresult == 1: poscount += 1
            else: cresult = 2

            tweets.append((cresult, result))

        pospercent = 0 if len(results) == 0 else "%.2f" \
                     % (float(poscount)*100/(poscount + negcount))
        negpercent = 0 if len(results) == 0 else "%.2f" \
                     % (float(negcount)*100/(poscount + negcount))

        self.set_header("Cache-Control","no-cache")
        self.render("index.html",
                    poscount = poscount,
                    negcount = negcount,
                    pospercent = pospercent,
                    negpercent = negpercent,
                    query = query, 
                    tweets = tweets)


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    settings = {
        "static_path" : os.path.join(dirname, "static"),
        "template_path" : os.path.join(dirname, "template")
    }
    application = tornado.web.Application([
        (r'/', MainHandler)
    ], **settings)

    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
    
