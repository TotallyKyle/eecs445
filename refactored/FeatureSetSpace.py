import itertools

class FeatureSetSpace:
  def __init__(self, possible_features, min_features = 1, required_keys = []):
    self._possibleFeatures  = possible_features
    self._minFeatures       = 1
    self._requiredKeys      = required_keys

  def __iter__(self):
    possible_feature_set = set(self._possibleFeatures) - set(self._requiredKeys)
    possible_feature_list = list(possible_feature_set)

    for i in range(max(0, self._minFeatures - len(self._requiredKeys)), len(possible_feature_list) + 1):
      for c in itertools.combinations(possible_feature_list, i):
        yield list(c) + self._requiredKeys
