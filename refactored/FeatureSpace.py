import itertools

class FeatureSetSpace:
  def __init__(self, possible_features):
    self._possibleFeatures            = possible_features

  def __iter__(self):
    for i in range(1, len(self._possibleFeatures) + 1):
      for c in itertools.combinations(self._possibleFeatures, i):
        yield c
