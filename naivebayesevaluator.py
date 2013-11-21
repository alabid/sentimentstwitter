#!/usr/bin/env python
'''
Contains implementation of Naive Bayes Classifier based upon
Generic Classifier for tweets.

Options:
--csvout -> put results in a CSV file
--stdout -> print results to STDOUT

Usage:
python naivebayesevaluator.py [(--csvout | --stdout) -g <grams-list> 
                              -w <weights-start, weights-end, step>
                              -t <thresholds-start, thresholds-end, step>]

For example, 
python naivebayesevaluator.py --csvout -g 1 1,2 -w 0.1,1.5,0.5 -t 1.0,2.0,0.5

should evaluate naive bayes using:
1. unigrams and unigrams + bigrams. 
2. weights 0.1, 0.6, 1.1
3. threshold values 1.0, 1.5, 2.0

and store the result in the file 'stats/nbevaluatorstats<current datetime>.csv'.
'''
from evaluator import Evaluator
from naivebayesclassifier import NaiveBayesClassifier
import csv
import datetime
import sys
import argparse

class NaiveBayesEvaluator(Evaluator):
    def __init__(self, trainfile, testfile, *args, **kargs):
        Evaluator.__init__(self, trainfile, testfile, *args, **kargs)

        self.allthresholds = kargs.get("allthresholds")
        self.csvout = kargs.get("csvout", False)
        self.results = []

    def flushToCSV(self):
        fname = "stats/nbevaluatorstats%s.csv" % str(datetime.datetime.now())
        with open(fname, "wb") as f:
            w = csv.writer(f, delimiter=',', quotechar='"')
            # write out header            
            w.writerow(["Classifier Info",
                        "Accuracy for Positives (%)",
                        "Accuracy for Negatives (%)",
                        "Accuracy for (Positives|Negatives) (%)",
                        "Correlation for (Positives|Negatives) (%)"])
            for row in self.results:
                w.writerow(row)
        print "Flushing results of Naive Bayes evaluation into '%s'..." % fname

    def run(self):
        for grams in self.allgrams:
            c = NaiveBayesClassifier(self.rawfname,
                                     grams=grams)
            c.trainClassifier()
            
            for w in self.allweights:
                c.setWeight(w)                                
        
                for t1 in self.allthresholds:
                    for t2 in self.allthresholds:
                        c.setThresholds(neg=t1, pos=t2)
                        cinfo, accpos, accneg, accall, corrall = self.evaluate(c)
                        self.results.append([cinfo, accpos, accneg,
                                             accall, corrall])

        if self.csvout:
            self.flushToCSV()

def processGrams(glist):
    return [[int(eachr) for eachr in each.split(',')] for each in glist]

def floatrange(start, end, step):
    return [start + step*x for x in range(int((end-start)/step)+1)]

def processWT(wstr):
    start, end, step = [float(res) for res in wstr.split(',')]
    return floatrange(start, end, step)

def main():
    trainfile = "trainingandtestdata/training.csv"
    testfile = "trainingandtestdata/testing.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--csvout", dest="csvout",
                        action="store_true", default=False)
    parser.add_argument("--stdout", dest="stdout", 
                        action="store_true", default=False)
    parser.add_argument("-g", dest="g", nargs="+",
                        metavar="x,y,z,..", required=True)
    parser.add_argument("-w", dest="w",
                        metavar="START, END, STEP", required=True)
    parser.add_argument("-t", dest="t",
                        metavar="START, END, STEP", required=True)

    args = parser.parse_args()
    grams = processGrams(args.g)        
    weights = processWT(args.w)
    thresholds = processWT(args.t)

    try:
        nbEvaluator = NaiveBayesEvaluator(trainfile, testfile,
                                          allgrams=grams,
                                          allweights=weights,
                                          allthresholds=thresholds,
                                          csvout=args.csvout,
                                          stdout=args.stdout)
        nbEvaluator.run()
    except:
        parser.print_help()

if __name__ == "__main__":
    main()
