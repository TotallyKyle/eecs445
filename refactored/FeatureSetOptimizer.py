from  FeatureSetSpace    import  FeatureSetSpace
from  ModelDataBuilder   import  ModelDataBuilder
import functools
import Queue

class FeatureSetOptimizer:

  def __init__(self, k = 1, min_features = 1):
    self._modelDataBuilder = ModelDataBuilder()
    self._k                = k
    self._minFeatures      = min_features
    self._optimalFeatures  = Queue.PriorityQueue()

  def AddDataColumn(self, data_column):
    self._modelDataBuilder.AddFeatureTimeSeries(data_column)

  def OptimizeFeatureSet(self, model):
    self._modelDataBuilder.BuildRawTimeSeries()
    for idx, feature_key_set in enumerate(FeatureSetSpace(self.FeatureKeys, self._minFeatures, self.RequiredKeys)):
      feature_set, output_set = self._modelDataBuilder.GenerateFeatures(
        functools.partial(self.RowFilterer, feature_key_set),
        functools.partial(self.RowFeatureMapper, feature_key_set),
        self.RowOutputMapper
      )

      error, extra_data = model(feature_set, output_set)
      print feature_key_set, error, extra_data

      try:
        self._optimalFeatures.put((error, feature_key_set, extra_data), False)
      except Queue.Full:
        pass   

    return FeatureSetOptimizer.ToList(self._optimalFeatures, self._k)

  @staticmethod
  def ToList(priority_queue, k):
    optimal_models = []
    i = 1
    while not priority_queue.empty() and i <= k:
      optimal_models.append(priority_queue.get_nowait())
      i += 1
    return optimal_models
