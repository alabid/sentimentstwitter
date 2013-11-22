

import cPickle as pickle
from evaluator import Evaluator
from maxentclassifier import MaximumEntropyClassifier

class MaxEntEvaluator(Evaluator):

    def __init__(self, trainfile, testfile, maxent_args = {}, **kargs):
        Evaluator.__init__(self, trainfile, testfile, **kargs)
        # TODO add dictionary for arguments to pass to MaximumEntropyClassifier
        self.maxent_args = maxent_args

    def run(self):
        ent = MaximumEntropyClassifier(self.rawfname, **self.maxent_args)
        print 'Initialized classifier, about to train...'
        ent.trainClassifier()

        self.evaluate(ent)

    def runFromPickle(self, picklefile):
      f = open(picklefile, "rb")
      ent = pickle.load(f)
      f.close()

      print 'Loaded classifier from', picklefile

      self.evaluate(ent)

def main():
    trainfile = "trainingandtestdata/training.csv"
    testfile = "trainingandtestdata/testing.csv"

    maxent_args = {
      'filesubset' : 3000,
      'min_occurences' : 3,
      'max_iter' : 4,
      'grams' : [1]
    }
    maxent_evaluator = MaxEntEvaluator(trainfile, 
                                       testfile,
                                       maxent_args,
                                       stdout = True
                                       )
    maxent_evaluator.run()
    #maxent_evaluator.runFromPickle('maxentpickles/maxent_100_1_2.dat')

if __name__ == '__main__':
  main()
