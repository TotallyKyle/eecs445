
class FeatureSetBuilder:
    def __init__(self):
        self._featureSet = {}

    def AddFeatureTimeSeries(self, feature_time_series):
        return False

    def IntersectAddedSeries(self):
        return False;

    def GetFeatureSet(self):
        return self._featureSet
