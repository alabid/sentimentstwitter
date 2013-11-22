# Contains implementation of a generic Classifier for tweets
import csv
import re
import nltk
import os
import cPickle as pickle
import random

class Classifier:
    # rawfname -> name of file containing raw training data
    # force == True iff user wants to overwrite classifier data
    # grams -> list of n-grams to use:
    #   1, unigrams; 2, bigrams; 3, trigrams; and so on
    #   so [1,2] means unigrams + bigrams
    # __init__(self, rawfname, modelfname, force, grams)
    def __init__(self, rawfname, *args, **kargs):
        self.rawfname = rawfname

        self.force = kargs.get("force", False)
        self.numgrams = kargs.get("grams", [1])
        
        # create modelfname using numgrams variable
        # e.g if self.numgrams = [1,2], then
        # self.modelfname = 'model1-2.dat'
        self.modelfname = "model%s.dat" % \
                          (reduce(lambda x, y: str(x)+'-'+str(y),
                                  self.numgrams))

        # weight to use in self.weightedProb
        self.weight = kargs.get("weight", 1.0)

        # The number of lines to train on. Use during development
        # to train on only a small chunk of the training set
        self.filesubset = kargs.get("filesubset", "all")

        # counts of tweets in each class
        # [x,y] where
        # x -> number of tweets in negative class
        # y -> number of tweets in positive class
        self.tweetcounts = [0, 0]

        # counts of feature/class combinations
        # stores (feature) => [x, y] where
        # x -> number of times feature appears in negative class
        # y -> number of times feature appears in positive class
        self.ftweetcounts = {}

    # Increment count of a feature/class pair
    def incFC(self, f, c):
        self.ftweetcounts.setdefault(f, [0, 0])
        self.ftweetcounts[f][c] += 1

    # Increment count of a class
    def incC(self, c):
        self.tweetcounts[c] += 1

    # Return number of times a features has appeared in a class
    def getFC(self, f, c):
        if f in self.ftweetcounts:
            return float(self.ftweetcounts[f][c])
        return 0.0

    # Return number of features in a class
    def getC(self, c):
        return float(self.tweetcounts[c])
    
    # Return total number of features 
    def getTotal(self):
        return sum(self.tweetcounts)

    # Each feature has weight 1
    # That is, even if the word 'obama' appears >10 times
    # in a tweet, it is counted only once in that particular tweet
    def getFeatures(self, item):
        flist = []
        for gram in self.numgrams:
            tokenized = nltk.word_tokenize(item)
            for i in range(len(tokenized)-gram+1):
                flist.append(" ".join(tokenized[i:i+gram]))
        return set(flist)

    # Train the classifier using item (for now, just text) on a specific class
    # c -> class (number)
    def train(self, c, item):
        features = self.getFeatures(item)
        for f in features:
            self.incFC(f, c)
        self.incC(c)

    # Trains the classifier based on tweets in self.modelfname
    # Stores the resulting data structures in a pickle file
    def trainClassifier(self):
        if self.force:
            os.remove(self.modelfname)        
        elif os.path.exists(self.modelfname):
            grams, self.tweetcounts, self.ftweetcounts = pickle.load(
                open(self.modelfname, "rb")
            )
            # stop iff we have data for the number of grams we want
            if grams == self.numgrams:
                print "Model retrieved from '%s'" % self.modelfname
                return

        f = open(self.rawfname)
        r = csv.reader(f, delimiter=',', quotechar='"')

        # get 0th column -> '0' if negative (class 0), '4' if positive (class 1)
        # get 5th column -> contains text of tweet
        stripped = [(0 if line[0] == '0' else 1, 
                     re.sub(r'[,.]', r'',
                            line[-1].lower().strip())) for line in r]

        # Only train on lines 0 -> <last_line> of the training set
        last_line = len(stripped) if self.filesubset == "all" else self.filesubset

        for each in stripped[:last_line]:
            self.train(each[0], each[1])

        # store Classifier training data
        pickle.dump([self.numgrams, self.tweetcounts, self.ftweetcounts],
                    open(self.modelfname, "wb")
        )

        print "Model stored in '%s'" % self.modelfname

        f.close()

    # Return <n> tweets from the training set where <pct_pos> of the tweets
    # have positive sentiment and (1 - <pct_pos>) have negative sentiment
    def getSampleTweets(self, n, pct_pos = .5):
        random.seed(10)
        numpos, numneg = 0, 0
        targetpos, targetneg = int(n * pct_pos), int(n * (1 - pct_pos))

        # Should have <n> lines in the end
        sample = []

        f = open(self.rawfname)
        r = csv.reader(f, delimiter=',', quotechar='"')

        # get 0th column -> '0' if negative (class 0), '4' if positive (class 1)
        # get 5th column -> contains text of tweet
        stripped = [(0 if line[0] == '0' else 1, 
                     re.sub(r'[,.]', r'',
                            line[-1].lower().strip())) for line in r]

        random.shuffle(stripped)
        
        i = 0

        # Read through the shuffled list of examples until there are 
        # <targetpos> positive tweets and <targetneg> negative tweets
        # in our sample
        while numpos < targetpos or numneg < targetneg:
            curtweet = stripped[i]

            if curtweet[0] == 0 and numneg < targetneg:
                numneg += 1
                sample.append(curtweet)
            elif curtweet[0] == 1 and numpos < targetpos:
                numpos += 1
                sample.append(curtweet)
            i += 1

        return sample
    # Return the probability of a feature being in a particular class
    def probFC(self, f, c):
        if self.getC(c) == 0: 
            return 0
        return self.getFC(f, c)/self.getC(c)

    # Return the probability Prob(Class)
    def probC(self, c):
        return self.getC(c)/self.getTotal()

    def setWeight(self, w):
        self.weight = w

    # Method of smoothing:
    # Start with an assumed probability (ap) for each word in each class
    # Then, return weighted probability of real probability (probFC)
    # and assumed probability
    # weight of 1.0 means ap is weighted as much as a word
    # Bayesian in nature: 
    # For example, the word 'dude' might not be in the corpus initially.
    # so P('dude' | class=0) = 0.5 and P('dude' | class=1) = 0.5
    # then when we find one 'dude' that's positive,
    # P('dude' | class=0) = 0.25 and P('dude' | class=1) = 0.75
    def weightedProb(self, f, c, ap=0.5):
        # calculate current probability
        real = self.probFC(f, c)
        
        # count number of times this feature has appeared in all categories
        totals = sum([self.getFC(f,c) for c in [0, 1]])
        
        # calculate weighted average
        return ((self.weight * ap) + (totals * real))/(self.weight + totals)

    # Return 0 if negative; Return 1 if positive
    def classify(self, text):
        raise Exception("You must subclass 'Classifier' to classify tweets")

    def __repr__(self):
        return "Classifier info: (weight=%s, grams=%s)" % (self.weight, self.numgrams)
