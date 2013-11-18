class Feature:

  def __init__(self, name, weight=None):
    self.name = name
    self.weight = weight

  def evaluate(self, intext):
    raise Exception("You must subclass 'Feature' to evaluate anything!")

class UnigramFeature(Feature):
  '''
  Feature type which simply returns 1 if the unigram is present, 0 otherwise
  '''

  def __init__(self, unigram, weight=None):
    self.unigram = unigram
    Feature.__init__(self, 'unigram_' + unigram, weight)

  def evaluate(self, intext):
    return 1 if self.unigram in intext else 0

