# Contains implementation of a generic Classifier for tweets
import csv
import re
import nltk
import os
import cPickle as pickle

class Classifier:
    # rawfname -> name of file containing raw training data
    # modelfname -> name of file to store classifier data
    # force == True iff user wants to overwrite classifier data
    # grams -> 1, unigrams; 2, bigrams; 3, trigrams; and so on
    # __init__(self, rawfname, modelfname, force, grams)
    def __init__(self, rawfname, *args, **kargs):
        self.rawfname = rawfname

        self.modelfname = kargs.get("modelfname", "model.dat")
        self.force = kargs.get("force", False)
        self.numgrams = kargs.get("grams", 1)

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
        tokenized = nltk.word_tokenize(item)
        flist = []
        for i in range(len(tokenized)-self.numgrams+1):
            flist.append(" ".join(tokenized[i:i+self.numgrams]))
        return set(flist)

    # Train the classifier using item (for now, just text) on a specific class
    # c -> class (number)
    def train(self, c, item):
        features = self.getFeatures(item)
        for f in features:
            self.incFC(f, c)
        self.incC(c)

    # Trains the classifier based on tweets in self.file
    # Stores the resulting data structures in a pickle file
    def trainClassifier(self):
        if self.force:
            os.remove(self.modelfname)        
        elif os.path.exists(self.modelfname):
            self.tweetcounts, self.ftweetcounts = pickle.load(
                open(self.modelfname, "rb")
            )
            return
            

        f = open(self.rawfname)
        r = csv.reader(f, delimiter=',', quotechar='"')

        # get 0th column -> '0' if positive (class 1), '4' if negative (class 0)
        # get 5th column -> contains text of tweet
        stripped = [(0 if line[0] == '4' else 1, 
                     re.sub(r'[,.]', r'',
                            line[-1].lower().strip())) for line in r]

        # Only train on lines 0 -> <last_line> of the training set
        last_line = len(stripped) if self.filesubset == "all" else self.filesubset

        for each in stripped[:last_line]:
            self.train(each[0], each[1])

        # store Classifier training data
        pickle.dump([self.tweetcounts, self.ftweetcounts],
                    open(self.modelfname, "wb")
        )

        f.close()

    # Return the probability of a feature being in a particular class
    def probFC(self, f, c):
        if self.getC(c) == 0: 
            return 0
        return self.getFC(f, c)/self.getC(c)

    # Return the probability Prob(Class)
    def probC(self, c):
        return self.getC(c)/self.getTotal()

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
    def weightedProb(self, f, c, weight=1.0, ap=0.5):
        # calculate current probability
        real = self.probFC(f, c)
        
        # count number of times this feature has appeared in all categories
        totals = sum([self.getFC(f,c) for c in [0, 1]])
        
        # calculate weighted average
        return ((weight * ap) + (totals + real))/(weight + totals)

    # Return 0 if negative; Return 1 if positive
    def classify(self, text):
        raise Exception("You must subclass 'Classifier' to classify tweets")

