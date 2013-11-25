'''
Used to evaluate the performance of the generic classifier
'''
import re
import csv

class Evaluator:
    '''
    trainfname => name of file containing raw training data
    testfname => name of file containing raw testing data
    force == True iff user wants to overwrite classifier data
    allgrams -> list of ALL n-grams to use:
               1, unigrams; 2, bigrams; 3, trigrams; and so on
               so [[1], [2]] means evaluate on unigram model, then bigrams
    allweights -> list of ALL weights (used in classifier.weightedProb) to use:
               [0.1, 1.0] means use weight=0.1, then weight=1.0
    '''
    def __init__(self, trainfname, devfname, testfname, *args, **kargs):
        self.usedev = kargs.get("usedev", False)

        if self.usedev:
            self.testdata = self.readTestData(devfname)
        else:
            self.testdata = self.readTestData(testfname)
            
        self.rawfname = trainfname

        self.allgrams = kargs.get("allgrams")
        self.allweights = kargs.get("allweights")
        # indicator variable to display evaluation results in STDOUT
        self.stdout = kargs.get("stdout", False)
        
    def evaluate(self, classifier):
        '''
        Returns some stats about how accurate classifier is on
        either the training set or the dev. set
        '''
        totalneg = 0
        totalpos = 0
        correctneg = 0
        correctpos = 0

        for test in self.testdata:
            # check if actual result of classifier classification matches
            # with expected result
            result = test[0]
            text = test[1]

            if result == 0:
                if classifier.classify(text) == 0:
                    correctneg += 1
                totalneg += 1
            elif result == 1:
                if classifier.classify(text) == 1:
                    correctpos += 1
                totalpos += 1
    
        correctall = correctpos + correctneg
        totalall = totalpos + totalneg

        # record accuracy, correlation
        accpos = float(correctpos)*100/totalpos
        accneg = float(correctneg)*100/totalneg
        accall = float(correctall)*100/totalall
        corrall = 100-float(abs(correctpos-correctneg))*100/totalall

        if self.stdout:
            print "="*100
            print classifier
            print "Accuracy for Positives: %.2f%%" % accpos
            print "Accuracy for Negatives: %.2f%%" % accneg
            print "Accuracy for (Positives|Negatives): %.2f%%" % accall
            print "Correlation for (Positives|Negatives): %.2f%%" % corrall
            print "="*100
            print

        return [str(classifier), accpos, accneg, accall, corrall]

    def readTestData(self, fname):
        testdata = []
        with open(fname) as f:
            r = csv.reader(f, delimiter=',', quotechar='"')
            for line in r:
                # get 0th column -> '0' if negative (class 0), '4' if positive (class 1)
                #                   '2' if neutral (class -1)
                # get 5th column -> contains text of tweet
                if line[0] == '0':
                    polarity = 0
                elif line[0] == '4':
                    polarity = 1
                else:
                    polarity = -1
                
                testdata.append([polarity,
                                 re.sub(r'[,.]', r'',
                                        line[-1].lower().strip())])
        return testdata

    def run(self):
        raise Exception("You must subclass 'Evaluator' and define run")
