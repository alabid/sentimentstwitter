# Implementation of Naive Bayes Classifier for tweets
import sys
import math
from classifier import Classifier

class MaxEntClassifier(Classifier):
    def __init__(self, fname, grams=[1], **kargs):
        Classifier.__init__(self, fname, grams, **kargs)

        # Dictionary {unigram -> <UnigramFeature> holding that <unigram>}
        self.features = self.initUnigramFeatures()

    
    def initUnigramFeatures(self):
        unigrams = self.ftweetcounts.keys()
        return {u : UnigramFeature(f) for u in unigrams}

    def classify(self, text):
        # TODO this does way more work than necessary by looping overa
        # all features. Just index features on their unigrams, set 
        # equal to 1 if unigram is in <text>, 0 if not
        unigrams = self.getFeatures(text)

        text_features = [self.features.get(u, None) for u in unigrams]
        
        odds_ratio = sum([f.weight for f in text_features if f != None])

        return 1 if odds_ratio > 0 else 0


def main():    
    # file to get training data from
    fromf = 'trainingandtestdata/training.csv'
    ent = MaxEntClassifier(fromf, filesubset = 1000)
    ent.trainClassifier()

    # optionally, pass in some tweet text to classify
    if len(sys.argv) == 2:
        print ent.classify(sys.argv[1])
    

if __name__ == "__main__":
    main()
