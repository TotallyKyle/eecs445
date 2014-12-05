
class FeatureSetBuilder:
    def __init__(self):
        self._featureSet = {}

    def AddFeature(self, feature_name):
        return False

    def AddTimeDelayedFeature(self, feature_name):
        return False

    def GetFeatureSet(self):
        return self._featureSet