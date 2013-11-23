# Implementation of Naive Bayes Classifier for tweets
import sys
import math
from classifier import Classifier

class NaiveBayesClassifier(Classifier):
    def __init__(self, fname, *args, **kargs):
        Classifier.__init__(self, fname, *args, **kargs)

        # sometimes a threshold value is trained during Bayesian
        # classification to avoid classifying too many 'documents' as
        # one kind or the other
        self.thresholds = [1.0, 1.0]

    def setThresholds(self, neg=1.0, pos=1.0):
        self.thresholds = [neg, pos]

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

    # Returns 0 (negative) if P(class=0 | tweet) > P(class=1 | tweet) * thresholds[0]
    # Return 1 (positive) if P(class=1 | tweet) > P(class=0 | tweet) * thresholds[1]
    # Else return -1 (neutral)
    def classify(self, text):
        p0 = self.probClassTweet(text, 0)
        p1 = self.probClassTweet(text, 1)

        if p0 > p1 + math.log(self.thresholds[0]):
            return 0
        elif p1 > p0 + math.log(self.thresholds[1]):
            return 1
        else:
            return -1

    def __repr__(self):
        return "Classifier info: (weight=%s, grams=%s, thresholds=%s)" % (self.weight, self.numgrams, self.thresholds)


def main():    
    # file to get training data from
    fromf = 'trainingandtestdata/training.csv'
    naive = NaiveBayesClassifier(fromf)
    naive.trainClassifier()

    # optionally, pass in some tweet text to classify
    if len(sys.argv) == 2:
        print
        text = sys.argv[1]
        result = naive.classify(text)
        if result == 0:
            print "'%s' predicted to be Negative" % text
        elif result == 1:
            print "'%s' predicted to be Positive" % text
        else:
            print "'%s' predicted to be Neutral" % text
    

if __name__ == "__main__":
    main()
