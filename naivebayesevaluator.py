# Contains implement of Naive Bayes Evaluator for tweets
# TODO: add options to print to std and/or store in csv
from evaluator import Evaluator
from naivebayesclassifier import NaiveBayesClassifier

class NaiveBayesEvaluator(Evaluator):
    def __init__(self, trainfile, testfile, *args, **kargs):
        Evaluator.__init__(self, trainfile, testfile, *args, **kargs)
        self.allthresholds = kargs.get("allthresholds")

    def run(self):
        bestscore = 0
        bestg = None
        bestw = None
        bestt1 = None
        bestt2 = None

        for grams in self.allgrams:
            c = NaiveBayesClassifier(self.rawfname,
                                     grams=grams)
            c.trainClassifier()
            
            for w in self.allweights:
                c.setWeight(w)                                
        
                for t1 in self.allthresholds:
                    for t2 in self.allthresholds:
                        c.setThresholds(neg=t1, pos=t2)
                        score = self.evaluate(c)

                        if bestscore < score:
                            bestscore = score
                            bestg, bestw, bestt1, bestt2 = grams, w, t1, t2
    
        print "Maximum accuracy achieved: %.2f%%" %  bestscore
        print "grams=%s, weight=%s, threshold=%s" % (bestg, bestw, [t1, t2])

def main():
    trainfile = "trainingandtestdata/training.csv"
    testfile = "trainingandtestdata/testing.csv"

    nbEvaluator = NaiveBayesEvaluator(trainfile, testfile,
                                      allgrams=[[1], [1, 2]],
                                      allweights=[0.1, 0.2, 0.3, 0.4, 0.5],
                                      allthresholds=[1, 10, 20])

    nbEvaluator.run()
    

if __name__ == "__main__":
    main()
