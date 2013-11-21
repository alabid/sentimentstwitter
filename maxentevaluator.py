
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


def main():
    trainfile = "trainingandtestdata/training.csv"
    testfile = "trainingandtestdata/testing.csv"

    maxent_args = {
      'filesubset' : 2000,
      'min_occurences' : 5,
      'max_iter' : 5,
    }
    maxent_evaluator = MaxEntEvaluator(trainfile, 
                                       testfile,
                                       maxent_args,
                                       min_occurences = 2,
                                       allgrams=[[1]],
                                       stdout = True
                                       )

    maxent_evaluator.run()

if __name__ == '__main__':
  main()
