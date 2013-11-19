# Contains implement of Naive Bayes Evaluator for tweets
from evaluator import Evaluator
from naivebayesclassifier import NaiveBayesClassifier

class NaiveBayesEvaluator(Evaluator):
    def __init__(self, trainfile, testfile, *args, **kargs):
        Evaluator.__init__(self, trainfile, testfile, *args, **kargs)
        self.allthresholds = kargs.get("allthresholds")

    def run(self):
        bestscore = 0
        bestclassifier = None
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
                            bestclassifier = c
    
        print "Maximum accuracy achieved: %.2f%%" %  bestscore
        print "Best classifier: "
        print bestclassifier

def main():
    trainfile = "trainingandtestdata/training.csv"
    testfile = "trainingandtestdata/testing.csv"

    nbEvaluator = NaiveBayesEvaluator(trainfile, testfile,
                                      allgrams=[[1]],
                                      allweights=[0.1, 0.5, 1.0, 1.5, 2.0],
                                      allthresholds=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    nbEvaluator.run()
    

if __name__ == "__main__":
    main()
