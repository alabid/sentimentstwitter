import cPickle as pickle
import os
import datetime
import csv

from evaluator import Evaluator
from maxentclassifier import MaximumEntropyClassifier

class MaxEntEvaluator(Evaluator):

    def __init__(self, trainfile, devfile, testfile, maxent_args = {}, **kargs):
        Evaluator.__init__(self, trainfile, devfile, testfile, **kargs)
        self.maxent_args = maxent_args

    def run(self):
        '''
        Trains a MaximumEntropyClassifier using <self.maxent_args> and evaluates
        the trained model
        '''
        ent = MaximumEntropyClassifier(self.rawfname, **self.maxent_args)
        print 'Initialized classifier, about to train...'
        ent.trainClassifier()

        self.evaluate(ent)

    def runFromPickle(self, picklefile):
      '''
      Opens the NLTK model stored in <picklefile> and uses that model for evaluation
      '''
      f = open(picklefile, "rb")
      # Pickle stores an NLTK model
      ent_model = pickle.load(f)
      f.close()

      print 'Loaded classifier from', picklefile
      ent = MaximumEntropyClassifier(self.rawfname, **self.maxent_args)
      ent.setModel(ent_model)

      # Return everything but the classifer string
      return self.evaluate(ent)[1:]
      

    def testAllPickles(self, pickledir='maxentpickles/'):
      '''
      Tests all models stored in pickles from <pickledir>
      '''
      pickle_files = os.listdir(pickledir)
      models = []

      for pick in pickle_files:
        print self.runFromPickle(pickledir + pick)
        accpos, accneg, accall, corrall = self.runFromPickle(pickledir + pick)
        
        models.append([pick, accpos, accneg, accall, corrall])

      self.flushToCSV(models)

    
    def flushToCSV(self, models, resultdir='maxentresults/'):
      '''
      Writes a file storing results from <models> to <resultdir>
      File is named the current time stamp 
      '''
      fname = resultdir + str(datetime.datetime.now()) + '.csv'

      with open(fname, "wb") as f:
        w = csv.writer(f, delimiter=',', quotechar='"')
            # write out header            
        w.writerow(["model",
                    "accpos",
                    "accneg",
                    "accall",
                    "corrall"])
        for row in models:
            w.writerow(row)

    def buildManyModels(self):
      '''
      Uses every combination of the parameters specified below to create a
      MaximumEntropyClassifier, train it, and evaluate it
      '''
      all_filesubsets = [2000, 4000, 6000]

      all_min_occurences = [3, 5, 7]
      max_iter = 4
      all_grams = [[1], [1,2]]

      for filesubset in all_filesubsets:
        for min_occurence in all_min_occurences:
          for grams in all_grams:
            self.maxent_args = {
              'filesubset' : filesubset,
              'min_occurences' : min_occurence,
              'max_iter' : max_iter,
              'grams' : grams
            }
            ent = MaximumEntropyClassifier(self.rawfname, **self.maxent_args)
            print 'About to train with', self.maxent_args
            ent.trainClassifier()
            self.evaluate(ent)



def main():
    trainfile = "trainingandtestdata/training.csv"
    devfile = "trainingandtestdata/devset.csv"
    testfile = "trainingandtestdata/testing.csv"

    maxent_args = {
      'filesubset' : 2000,
      'min_occurences' : 4,
      'max_iter' : 4,
      'grams' : [1, 2]
    }
    
    # <stdout> controls whether or not to show output/progress
    # <usedev> == True to use the dev. set for evaluation, otherwise use the test set
    maxent_evaluator = MaxEntEvaluator(trainfile,
                                       devfile, 
                                       testfile,
                                       maxent_args,
                                       stdout = True,
                                       usedev = False
                                       )
    
    # Could run:
    # (1) To test all pickled models: maxent_evaluator.testAllPickles()
    # (2) To create/read one cached one model and evaluate it: maxent_evaluator.run()
    # (3) To build a ton of models: maxent_evaluator.buildManyModels()
    # This will take a LONG time, and parameters should be tweaked within the method
    #maxent_evaluator.run()

    # Uncomment the line below to run the better, pickled version
    maxent_evaluator.runFromPickle('maxentpickles/maxent_4000_3_2.dat')
    

if __name__ == '__main__':
  main()
