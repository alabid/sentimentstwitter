Sentiment Analysis on Twitter
=============================

### Screenshots of app
* [Tweets] (http://blog.alabidan.me/?attachment_id=382)
* [Donut Chart](http://blog.alabidan.me/?attachment_id=383)

### The  Problem
Given a tweet (that contains some text), estimate the sentiment
(negative or positive) of the tweeter.

### Why Hard?  

* Twitter allows only 140 characters -- not always enough to express
clear opinions.
* Accurate sarcasm detection is hard, but often present in
tweets.
* No spell checking, so the "same" word may be spelled differently.
For example, "love" and "loooooove" should have the same meaning, as should
"ur" and "your." 
* Hashtags are useful only when you have enough data that is
specific to a particular period. 
Our dataset came from Apr. 06 to Jun. 06 and therefore doesn't 
include recent topics.
  
### Features indicative of Sentiment in a Tweet
 
* Emoticons: :) :-) :( :D (but only if not misused!)
* Hashtags: Could possibly be useful, but are 
mostly relevant at a particular time (they could be
gone after hours, minutes, or even seconds).
* Words: Words are the most consistent features, but can still
be problematic due to lots of mispellings and abbreviations.
  
Training, Development, and Test Datasets}
Some folks at Stanford spent more than a year doing
research on sentiment analysis on twitter. They published a 
paper [here] (http://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)
and released both their training and test sets, which we used
throughout our project. 
  
The training set has 1,600,000 tweets marked positive/negative, while
the test set has 498 tweets. We extracted a development set of 
500 tweets from the original training set to use in adjusting 
various parameters for both types of classifiers.
  
### Method 1: Naive Bayes Classifier
Using a Naive Bayes Classifier, we can
estimate P(class=0 | tweet) and P(class=1 | tweet).
The tweet would be classified as **positive** if 
 P(class=1 | tweet) > threshold * P(class=0 | tweet). 
Otherwise, it would be classified as negative. 
The threshold value used here can be trained. For a more
detailed explanation of the Naive Bayes classifier, see
[this page] (http://en.wikipedia.org/wiki/Naive_Bayes_classifier).
  
Naive Bayes assumes that features are independent.
Despite the theoretical "shortcomings" of the Naive Bayes
classifier, it surprisingly performs pretty well in classifying
items into pre-specified classes
using features not necessarily independent. 

Our implementation of Naive Bayes depends on the following
parameters:
  * **Weight and Assumed Probability:** Both parameters are used as a 
    method of smoothing.
    For every word feature, we start with an assumed
    probability for each class.
    For example, the word 'dude' might not be in the corpus
    initially. So assuming weight of 1.0, then
    P('dude' | class=0) = 0.5 and 
    P('dude' | class=1) = 0.5 then 
    when we find one 'dude' that's positive,
    P('dude' | class=0) = 0.25 and 
    P('dude' | class=1) = 0.75

  * **Word feature types:** We tried our Naive Bayes
    classifier model on both unigram and unigram + bigram
    language models.
     
  * **Thresholds for sentiment classification**:
     We train two thresholds, one for the negative class and
     one for the positive class.

We trained the naive bayes classifier (on the dev. set) using weight range
[0.000005,0.1] and
threshold range [0.5, 10.0]
while testing on unigrams standalone and unigrams + bigrams. 

We saw a max accuracy of 78.8% on the dev. set. 
Using the "best"
parameters on our naive bayes classifier, we achieved an accuracy
of 83.01%. It turns out that our Naive Bayes classifier
performs better than that of Alec Go's. The Naive Bayes
classifier is implemented in 
**naivebayesclassifier.py** and the Naive Bayes
evaluator (which measures the effectiveness of the 
classifier) is implemented in 
**naivebayesevaluator.py**. 
  
Run `python naivebayesevaluator.py -g 1 1,2` to see
the accuracy on the test set. 
  
Run `python naivebayesevaluator.py -h` to see all 
options for the Naive Bayes Evaluator. For more details, see 
documentation in the evaluator file.
  
### Method 2: Maximum Entropy Classifier
  A maximum entropy classifier is a model which assigns weights to a set of
  features and classifies an example based on its particular features.
  The classification task is general to **k** classes, but we limit
  ourselves to classifying a tweet as positive or negative (leaving
  **k = 2**). 
  
  The basic idea of logistic regression requires that each tweet have
  a set of features. For text classification, these features are
  generally binary indicators for **n**-grams. In other words, the text's
  features are 0 for any **n**-gram not found in the text and 1 for
  any **n**-gram in the text. We use unigrams and bigrams
  as our features. This creates a number of practical limitations --
  most notably, using all unigrams/bigrams from a corpus creates 
  an unfeasible number of features. Instead, as mentioned by
  Pang et. al., we only use features seen more than some threshold number
  of times (generally between 3 and 5 -- note that this is one of 
  many parameters which can be tweaked). 
  
  Furthermore, training a logistic regression model is a computationally
  expensive action. As a result, we didn't use the entire training
  set to train the maximum entropy classifier. Instead, we used 
  between 2000 and 6000 randomly selected tweets (another parameter). 
  
  Therefore, the input to an instance of the Maximum Entropy Evaluator
  is made up of four parameters:
    * Number of tweets to train on (**filesubset**)
    * Minimum number of occurences a feature must have appeared
    to be included as a feature (**min_occurences**)
    * The number of iterations to run GIS (**max_iter**)
    * What **n**-grams to use (**grams**, a list)
  
  These parameters can be tweaked in **maxentevaluator**. Generally,
  we noted the intuitive trend that using more data gave better results.
  As a result, to get the best results, we would use a large subset
  of tweets and a small threshold  level of feature occurences 
  (i.e. including as many **n**-grams as possible). For practical purposes
  of demonstration, the parameters are set to lower values which only take
  a few minutes to run. 
  
  In order to use a better model (which achieves 76% on the test set),
  we've included a pickled model
  which was trained using 4000 tweets, unigrams and bigrams, and 
  a threshold of 3. When a Maximum Entropy Classifier is trained, the 
  resulting model is pickled to **maxentpickles**. While we
  have not included all the pickled models, the file 
  **maxent_4000_3_2.dat** is included and can be run via 
  **maxentevaluator**'s **runFromPickle** method. 
  

  
## Web Interface
  Our models were trained and evaluated
  on data accumulated in 2006. We thought it might be useful
  (and cool)
  to evaluate our models (Naive Bayes and Max Entropy)
  on more recent tweets using the 
  Twitter real-time API. So we setup a python Tornado
  server with a Graphical User Interface (built using
  HTML + CSS + Javascript) to grab tweets, perform
  sentiment analysis on these tweets, and display
  the results in an intuitive manner. 
  
  If you want to run the web server, you need
  tornado web (which can be easily intalled via 
  **pip** or **easy_install**. Use `python app.py` to startup the server. The file
  first loads both classifiers (takes about 2 minutes to load
  the models). Wait
  till you see the messages: `Model retrieved from 'model1-2.dat'` and `Max ent model built`. By 
  default, it runs on port 8888. To see the server in 
  action, using your favorite browser go to
  `http://localhost:8888/`. Pick what model you want
  to evaluate the tweet on, and **Search**.
  

MIT Open Source License
-----------------------
Copyright © 2012 Daniel Alabi, Nick Jones

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.