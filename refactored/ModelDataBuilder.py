
class ModelDataBuilder:

  @staticmethod
  def Normalize(unnormalized_lists):
    normalized_lists = []
    ranges           = []
    for l in unnormalized_lists:
      normalized_lists.append([])

    for idx in range(0, len(unnormalized_lists[0])):
      temp = []
      for l in unnormalized_lists:
        temp.append(l[idx])

      min_val = min(temp)
      max_val = max(temp)

      ranges.append([min_val, max_val]);

      for i, l in enumerate(unnormalized_lists):
        normalized_target = (l[idx] - min_val) / (max_val - min_val)
        normalized_lists[i].append(normalized_target)
                
    return normalized_lists, ranges

  @staticmethod
  def Denormalize(normalized_lists, ranges):
    unnormalized_lists = []

    for row in normalized_lists:
      unnormalized_row = []
      for idx, v in enumerate(row):
        min_val, max_val = ranges[idx]
        denormalized_value = \
          (v * (max_val - min_val)) + min_val
        unnormalized_row.append(denormalized_value)
      unnormalized_lists.append(unnormalized_row)
    return unnormalized_lists

  def __init__(self, train = .7, validation = 0, test = .3):
    self._train       = train
    self._validation  = validation
    self._test        = test
    self._series      = {}
    self._features    = None
    self._outputs     = None
  
  def AddFeatureTimeSeries(self, time_series):
    self._series[time_series.FeatureName] = time_series

  def BuildRawTimeSeries(self):
    date_set = self._IntersectedTimeSeriesDates()
    data     = []

    for idx, date in enumerate(date_set):
      features = {}
      for feature, series in self._series.iteritems():
        # We know that date is in all three sets
        features[feature] = series.Data[date]

      data.append(features)

    # Returns an array of dictionaries of feature_name to value
    self._dates   = list(date_set)
    self._rawData = data

  def GetRawData(self):
    data = self._rawData
    return data
  #
  #
  #
  #
  #
  def GenerateFeatures(self, filter_func, map_input_func, map_output_func):
    features = []
    outputs  = []
    ranges   = {}
    self.filter_func = filter_func
    num_rows = len(self._rawData)
    for idx, row in enumerate(self._rawData):
      date = self._dates[idx]
      if filter_func != None:
        if filter_func(idx, num_rows, date) == False:
          continue
      # Map input row into new features if passed function
      input_vals = row
      if map_input_func != None:
        input_vals = map_input_func(idx, self._rawData)
      features.append(input_vals)

      # Map raw data to output values
      outputs.append(map_output_func(idx, self._rawData))

      # Handle ranges for input data
      for feature, value in input_vals.iteritems():
        if feature not in ranges:
          ranges[feature] = [value, value]
        else:
          vals = ranges[feature]
          if value < vals[0]:
            # new min
            vals[0] = value
          if value > vals[1]:
            # new max
            vals[1] = value
    self._rawRange = ranges
    # Don't actually modify the raw data
    return features, outputs

  def MapInputToArrays(self):
    mapped_input_data = []
    for idx, row in enumerate(self._rawData):
      date = self._dates[idx]
      if self.filter_func != None:
        if self.filter_func(idx, len(self._rawData), date) == False:
          continue
      mapped_input_data.append(self._DefaultMapInputFunction(row))
    return mapped_input_data

  def MapOutputToArrays(self, map_output_func, target_filter_func = None):
    mapped_output_data = []
    num_rows = len(self._rawData)
    for idx, row in enumerate(self._rawData):
      date = self._dates[idx]
      if target_filter_func != None:
        if target_filter_func(idx, num_rows, date) == False:
          continue
      output_vals = map_output_func(idx, self._rawData)
      mapped_output_data.append(self._DefaultMapTargetFunction(output_vals))
    return mapped_output_data

  def MapRangesToArray(self):
    mapped_ranges = [self._rawRange[key] for key in sorted(self._rawRange.keys())]
    return mapped_ranges

  def MapToArrays(self, map_output_func, filter_func = None, map_input_func = None):
    # Map array of dictionaries to array of lists
    mapped_input_data  = []
    mapped_output_data = []
    ranges             = {}

    num_rows = len(self._rawData)

    # Assumes that raw data has been built
    for idx, row in enumerate(self._rawData):
      date = self._dates[idx]

      # Filter based on passed function
      if filter_func != None:
        if filter_func(idx, num_rows, date) == False:
          continue

      # Map input row into new features if passed function
      input_vals = row
      if map_input_func != None:
        input_vals = map_input_func(idx, self._rawData)

      # Handle ranges for input data
      for feature, value in input_vals.iteritems():
        if feature not in ranges:
          ranges[feature] = [value, value]
        else:
          vals = ranges[feature]
          if value < vals[0]:
            # new min
            vals[0] = value
          if value > vals[1]:
            # new max
            vals[1] = value

      mapped_input_data.append(self._DefaultMapInputFunction(input_vals))


      # Map data into outputs
      output_vals = map_output_func(idx, self._rawData)
      mapped_output_data.append(self._DefaultMapTargetFunction(output_vals))

    # Map ranges to 2D array
    mapped_ranges = [ranges[key] for key in sorted(ranges.keys())]
    return mapped_input_data, mapped_output_data, mapped_ranges

  def MapBackToFeatures(self, data):
    mapped_data = []
    for row in data:
      mapped_row = {}
      for idx, col in enumerate(row):
        mapped_row[self._features[idx]] = col
      mapped_data.append(mapped_row)
    return mapped_data

  def MapBackToOutputs(self, data):
    mapped_data = []
    for row in data:
      mapped_row = {}
      for idx, col in enumerate(row):
        mapped_row[self._outputs[idx]] = col
      mapped_data.append(mapped_row)
    return mapped_data

  def _IntersectedTimeSeriesDates(self):
    iterator = self._series.itervalues()
    times_set = iterator.next().TimesSet()
    
    for series in iterator:
      times_set.intersection_update(series.TimesSet())

    return sorted(times_set)

  def _DefaultMapInputFunction(self, row):
    if self._features == None:
      self._features = sorted(row.keys())

    return [row[key] for key in self._features]

  def _DefaultMapTargetFunction(self, row):
    if self._outputs == None:
      self._outputs = sorted(row.keys())

    return [row[key] for key in self._outputs]
