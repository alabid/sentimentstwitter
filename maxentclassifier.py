# Implementation of Naive Bayes Classifier for tweets
import sys
import csv
import re
import os
import cPickle as pickle

from classifier import Classifier
from evaluator import Evaluator
from nltk.classify.maxent import MaxentClassifier

class MaximumEntropyClassifier(Classifier):
    def __init__(self, rawfname, min_occurences=1, **kargs):
        Classifier.__init__(self, rawfname, **kargs)

        self.min_occurences = min_occurences

        # Maintains all training examples
        self.all_training_examples = []

        # Each example contains only keys for features which occurred more
        # than <min_occurences> times in the training set
        self.shrunk_training_examples = []

        # { feature -> num times <feature> was seen }
        self.all_features = {}
        self.model = None

        self.filesubset = kargs.get('filesubset', 'all')

        self.max_iter = kargs.get('max_iter', None)

        # Terminate if a single iteration improves log likelihood by less than v
        # (an argument passed to the max ent classifier)
        self.min_lldelta = kargs.get('min_lldelta', None)

    
    def setModel(self, model):
        self.model = model


    def initFeatures(self):
        '''
        Grabs a sample of size <self.filesubet> (half of which are pos., half neg.)
        and extracts features for each example. Note that features are 1, 2, ... grams
        based on <self.numgrams>

        Since there are far too many features using all uni/bi/...grams, only features
        that were seen more than <self.min_occurences> times (over all examples) are kept.
        Feature vectors are then padded with 0s for each feature that didn't appear in 
        a given training example
        '''

        training_sample = self.getSampleTweets(self.filesubset)
        print 'Training on %i lines' % len(training_sample)

        for each in training_sample:
            classification = each[0]
            text = each[1]
            feature_set = self.getFeatures(text)

            feature_vector = self.getFeatureDict(feature_set)
            # TODO need to pad the feature vectors with 0s?
            self.all_training_examples.append((feature_vector, classification))       

        shrunk_features = self.shrinkFeatureSet()
        print 'Padding feature vectors. There are %i total features' % len(self.all_features)
        self.initShrunkExamples(shrunk_features)
        print 'Padding done'
        
    
    def shrinkFeatureSet(self):
        shrunk = {}

        # TODO make this a dict-comprehension
        for feat, num_ocurrences in self.all_features.iteritems():
            if num_ocurrences >= self.min_occurences:
                shrunk[feat] = num_ocurrences

        print 'Shrunk down to %i features' % len(shrunk)

        return shrunk


    def initShrunkExamples(self, shrunk_features):
        '''
        Set up <self.shrunk_training_examples>, a list of tuples of the form:
        ({features}, classification) where {features} is made up of all features in <shrunk_features>
        and a 1 if the given feature was in a given example
        '''

        # Set <self.shrunk_training_examples> to include only features in <shrunk_features>
        # with a 1 indicating presence of a feature, 0 otherwise
        for i in range(len(self.all_training_examples)):
            example = self.all_training_examples[i]
            shrunk_feature_vector = {}

            for feat in shrunk_features:
                shrunk_feature_vector[feat] = 1 if feat in example[0] else 0 

            self.shrunk_training_examples.append((shrunk_feature_vector, example[1]))


    def getFeatureDict(self, featureset):
        '''
        Update <self.all_features> to include the new features seen, and return a 
        dictionary containing '1' for each of those features
        '''
        feature_dict = {}

        for feat in featureset:
            feature_dict[feat] = 1
            self.all_features.setdefault(feat, 0)
            self.all_features[feat] += 1
        
        return feature_dict


    def trainClassifier(self):
        '''
        Calculates features and trains the maxent classifier, storing the resulting
        model in <self.model>
        '''
        # check if pickled
        pickled_model = self.checkForPickle()
        if pickled_model:
            self.model = pickled_model
        else:

            self.initFeatures()
            print 'Done reading in training examples'
            kargs = {
                'algorithm' : 'gis',
            }
            if self.max_iter != None:
                kargs['max_iter'] = self.max_iter

            self.model = MaxentClassifier.train(self.shrunk_training_examples, **kargs)
            self.pickleModel()
        print 'Max ent model built'


    def classify(self, text):
        feature_set = self.getFeatures(text)
        feature_vector = self.getFeatureDict(feature_set)

        return self.model.classify(feature_vector)

    def checkForPickle(self):
        pickle_name = self.getPickleFileName()

        if os.path.exists(pickle_name):
            f = file(pickle_name, 'rb')
            model = pickle.load(f)
            f.close()

            return model
        else:
            return False

    def pickleModel(self, model_name=None):
        '''
        Saves the current Classifier object in a file called:
        "maxent_<len of subset used>_<min num a feature was seen>_<number of grams used>"
        Note that every model uses increasing number of n-grams, so the length tells us 
        the max n-gram (i.e. length of 2 indicates we used unigrams and bigrams while length
        of 1 indicates we only used unigrams)
        '''
        if model_name == None:
            model_name = 'maxentpickles/maxent_%i_%i_%i.dat' % \
                         (self.filesubset, self.min_occurences, len(self.numgrams))

        outfile = open(model_name, "wb")
        pickle.dump(self.model, outfile)

        outfile.close()

    def getPickleFileName(self):
        return 'maxentpickles/maxent_%i_%i_%i.dat' % \
               (self.filesubset, self.min_occurences, len(self.numgrams))





def main():    
    # file to get training data from
    trainfile = "trainingandtestdata/training.csv"
    testfile = "trainingandtestdata/testing.csv"

    maxent_args = {
      'filesubset' : 3500,
      'min_occurences' : 5,
      'max_iter' : 4,
      'grams' : [1]
    }
    ent = MaximumEntropyClassifier(fromf, filesubset = 500, max_iter = 20)
    ent.trainClassifier()

    # optionally, pass in some tweet text to classify
    if len(sys.argv) == 2:
        print ent.classify(sys.argv[1])
    

if __name__ == "__main__":
    main()
