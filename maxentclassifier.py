# Implementation of Naive Bayes Classifier for tweets
import sys
import csv
import re
import math
import random
from classifier import Classifier
from evaluator import Evaluator
from nltk.classify.maxent import MaxentClassifier

class MaximumEntropyClassifier(Classifier):
    def __init__(self, fname, grams=[1], **kargs):
        self.fname = fname
        self.numgrams = grams
        self.training_examples = []
        self.all_features = set() 
        self.model = None

        self.filesubset = kargs.get('filesubset', 'all')

        self.max_iter = kargs.get('max_iter', None)

    
    def initUnigramFeatures(self):

        f = open(self.fname)
        r = csv.reader(f, delimiter=',', quotechar='"')

        # get 0th column -> '0' if negative (class 0), '4' if positive (class 1)
        # get 5th column -> contains text of tweet
        stripped = [(0 if line[0] == '0' else 1, 
                     re.sub(r'[,.]', r'',
                            line[-1].lower().strip())) for line in r]

        # Shuffle the lines so we get <last_line> random lines
        random.shuffle(stripped)

        # Only train on lines 0 -> <last_line> of the training set
        last_line = len(stripped) if self.filesubset == "all" else self.filesubset
        print 'Training on %i lines' % last_line

        for each in stripped[:last_line]:
            classification = each[0]
            text = each[1]
            feature_set = self.getFeatures(text)

            feature_vector = self.getFeatureDict(feature_set)
            # TODO need to pad the feature vectors with 0s?
            self.training_examples.append((feature_vector, classification))       

        print 'Padding feature vectors. There are %i total features' % len(self.all_features)
        self.padFeatureVectors()
        print 'Padding done'
        
    
    # Add 0 entries for all features which a given <training_example> didn't have
    def padFeatureVectors(self):
        for example in self.training_examples:
            for feat in self.all_features:
                if feat not in example[0]:
                    example[0][feat] = 0




    # Update <self.all_features> to include the new features seen, and return a 
    # dictionary containing '1' for each of those features
    def getFeatureDict(self, featureset):
        self.all_features.update(featureset)

        return {f : 1 for f in featureset}


    def trainClassifier(self):
        self.initUnigramFeatures()
        print 'Done reading in training examples'
        kargs = {
            'algorithm' : 'iis',
        }
        if self.max_iter != None:
            kargs['max_iter'] = self.max_iter

        self.model = MaxentClassifier.train(self.training_examples, **kargs)
        print 'Max ent model built'


    def classify(self, text):
        feature_set = self.getFeatures(text)
        feature_vector = self.getFeatureDict(feature_set)

        return self.model.classify(feature_vector)


def main():    
    # file to get training data from
    fromf = 'trainingandtestdata/training.csv'
    ent = MaximumEntropyClassifier(fromf, filesubset = 50)
    ent.trainClassifier()

    # optionally, pass in some tweet text to classify
    if len(sys.argv) == 2:
        print ent.classify(sys.argv[1])
    

if __name__ == "__main__":
    main()
