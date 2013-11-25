'''
Contains main implementation of a generic Classifier for tweets.
Reads in language model and stores into a model.
'''
import csv
import re
import nltk
import os
import cPickle as pickle
import random

class Classifier:
    '''
    rawfname -> name of file containing raw training data
    force == True iff user wants to overwrite classifier data
    grams -> list of n-grams to use:
             1, unigrams; 2, bigrams; 3, trigrams; and so on
             so [1,2] means unigrams + bigrams
    __init__(self, rawfname, modelfname, force, grams)
    '''
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
        self.weight = kargs.get("weight", 0.00005)

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

    def incFC(self, f, c):
        '''
        Increment count of a feature/class pair
        '''
        self.ftweetcounts.setdefault(f, [0, 0])
        self.ftweetcounts[f][c] += 1

    def incC(self, c):
        '''        
        Increment count of a class
        '''
        self.tweetcounts[c] += 1

    def getFC(self, f, c):
        '''
        Return number of times a features has appeared in a class
        '''
        if f in self.ftweetcounts:
            return float(self.ftweetcounts[f][c])
        return 0.0

    def getC(self, c):
        '''
        Return number of features in a class
        '''
        return float(self.tweetcounts[c])
    
    def getTotal(self):
        '''
        Return total number of features 
        '''
        return sum(self.tweetcounts)

    def getFeatures(self, item):
        '''
        Each feature has weight 1
        That is, even if the word 'obama' appears >10 times
        in a tweet, it is counted only once in that particular tweet
        '''
        flist = []
        for gram in self.numgrams:
            tokenized = nltk.word_tokenize(item)
            for i in range(len(tokenized)-gram+1):
                flist.append(" ".join(tokenized[i:i+gram]))
        return set(flist)

    def train(self, c, item):
        '''
        Trains the classifier using item (for now, just text) on a 
        specific class
        c -> class (number)
        '''
    
        features = self.getFeatures(item)
        for f in features:
            self.incFC(f, c)
        self.incC(c)

    def trainClassifier(self):
        '''
        Trains the classifier based on tweets in self.modelfname
        Stores the resulting data structures in a pickle file
        '''
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

    def getSampleTweets(self, n, pct_pos = .5):
        '''
        Return <n> tweets from the training set where <pct_pos> of the tweets
        have positive sentiment and (1 - <pct_pos>) have negative sentiment
        '''
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

    def probFC(self, f, c):
        '''
        Return the probability of a feature being in a particular class
        '''
        if self.getC(c) == 0: 
            return 0
        return self.getFC(f, c)/self.getC(c)

    def probC(self, c):
        '''
        Return the probability Prob(Class)
        '''
        return self.getC(c)/self.getTotal()

    def setWeight(self, w):
        '''
        Set weight to use in classifier
        '''
        self.weight = w

    def weightedProb(self, f, c, ap=0.5):
        '''
        Method of smoothing:
        Start with an assumed probability (ap) for each word in each class
        Then, return weighted probability of real probability (probFC)
        and assumed probability
        weight of 1.0 means ap is weighted as much as a word
        Bayesian in nature: 
        For example, the word 'dude' might not be in the corpus initially.
        so assuming weight of 1.0, then
        P('dude' | class=0) = 0.5 and P('dude' | class=1) = 0.5
        then when we find one 'dude' that's positive,
        P('dude' | class=0) = 0.25 and P('dude' | class=1) = 0.75
        '''
        # calculate current probability
        real = self.probFC(f, c)
        
        # count number of times this feature has appeared in all categories
        totals = sum([self.getFC(f,c) for c in [0, 1]])
        
        # calculate weighted average
        return ((self.weight * ap) + (totals * real))/(self.weight + totals)


    def classify(self, text):
        '''
        Return 0 if negative; Return 1 if positive
        '''
        raise Exception("You must subclass 'Classifier' to classify tweets")

    def __repr__(self):
        return "Classifier info: (weight=%s, grams=%s)" % (self.weight, self.numgrams)
