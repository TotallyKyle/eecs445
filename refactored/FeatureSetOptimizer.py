from  FeatureSetSpace    import  FeatureSetSpace
from  ModelDataBuilder   import  ModelDataBuilder
import functools

class FeatureSetOptimizer:

  def __init__(self, k = 1):
    self._modelDataBuilder = ModelDataBuilder()

  def AddDataColumn(self, data_column):
    self._modelDataBuilder.AddFeatureTimeSeries(data_column)

  def OptimizeFeatureSet(self, training_fn = None):
    self._modelDataBuilder.BuildRawTimeSeries()
    for feature_set in enumerate(FeatureSetSpace(self.FeatureKeys)):
      [inputs, targets, feature_ranges] = self._modelDataBuilder.MapToArrays(
        self.RowOutputMapper,
        functools.partial(self.RowFilterer, feature_set),
        functools.partial(self.RowFeatureMapper, feature_set)
      )

