#! /usr/bin/env python
import tornado.ioloop
import tornado.web
import urllib

from hidden import *

import tweepy

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        query = self.get_argument("query", "")
        
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        
        api = tweepy.API(auth)

        results = api.search(q=urllib.quote(query))

        for result in results:
            self.write("<p>" + result.text + "</p>")

application = tornado.web.Application([
    (r'/', MainHandler)
])

if __name__ == "__main__":
    application.listen(80)
    tornado.ioloop.IOLoop.instance().start()
