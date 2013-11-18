# Implementation of Naive Bayes Classifier for tweets
import sys
import math
from classifier import Classifier

class NaiveBayesClassifier(Classifier):
    def __init__(self, fname, grams=1, **kargs):
        Classifier.__init__(self, fname, grams, **kargs)

        # sometimes a threshold value is trained during Bayesian
        # classification to avoid classifying too many 'documents' as
        # negative
        self.threshold = 1

    # Returns the (log) probability of a tweet, given a particular class
    # P(tweet | class)
    def probTweetClass(self, text, c):
        features = self.getFeatures(text)
        p = 0
        for f in features:
            p += math.log(self.weightedProb(f, c))
        return p

    # Returns the (log) probability of a class, given a particular tweet
    # P(class | tweet) = P(tweet | class) x P(class) / P(tweet)
    # But P(tweet) is constant for all classes; so forget
    def probClassTweet(self, text, c):
        return self.probTweetClass(text, c) + math.log(self.probC(c))

    # Returns 0 if P(tweet | class=0) > P(tweet | class=1) * threshold
    # Return 1 otherwise
    def classify(self, text):
        p0 = self.probClassTweet(text, 0)
        p1 = self.probClassTweet(text, 1)

        if p0 > p1 * self.threshold:
            return 0
        else:
            return 1

def main():    
    # file to get training data from
    fromf = 'trainingandtestdata/training.csv'
    naive = NaiveBayesClassifier(fromf, filesubset = 1000)
    naive.trainClassifier()

    # optionally, pass in some tweet text to classify
    if len(sys.argv) == 2:
        print naive.classify(sys.argv[1])
    

if __name__ == "__main__":
    main()
