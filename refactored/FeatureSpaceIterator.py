

class FeatureSpace

  def __init__(self, possible_features):
    self._possibleFeatures           = possible_features;
    self._combinationValue           = 0;
    self._currentCombination         = []
    self._currentCombinationIterator = self._currentCombination.__iter__();

  def __iter__(self):
