
class ModelDataBuilder:

  @staticmethod
  def Normalize(unnormalized_lists):
    normalized_lists = []
    for idx in range(0, len(unnormalized_lists[0])):
      temp = []
      for l in unnormalized_lists:
        normalized_lists.append([])
        temp.append(l[idx])

      min_val = min(temp)
      max_val = max(temp)

      for i, l in enumerate(unnormalized_lists):
        normalized_target = (l[idx] - min_val) / (max_val - min_val)
        normalized_lists[i].append(normalized_target)
                

    return normalized_lists

  def __init__(self, train = .7, validation = 0, test = .3):
    self._train      = train
    self._validation = validation
    self._test       = test
    self._series     = {}
  
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

  #
  #
  #
  #
  #
  def MapToArrays(self, map_output_func, filter_func = None, map_input_func = None):
    # Map array of dictionaries to array of lists
    mapped_input_data  = []
    mapped_output_data = []
    ranges             = {}

    # Assumes that raw data has been built
    for idx, row in enumerate(self._rawData):
      date = self._dates[idx]

      # Filter based on passed function
      if filter_func != None:
        if filter_func(row, idx, self._rawData, date) == False:
          continue

      # Map input row into new features if passed function
      input_vals = row
      if map_input_func != None:
        input_vals = map_input_func(row, idx, self._rawData, date)

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
      output_vals = map_output_func(row, idx, self._rawData, date)
      mapped_output_data.append(output_vals)


    # Map ranges to 2D array
    mapped_ranges = [ranges[key] for key in sorted(ranges.keys())]
    return mapped_input_data, mapped_output_data, mapped_ranges


  def _IntersectedTimeSeriesDates(self):
    iterator = self._series.itervalues()
    times_set = iterator.next().TimesSet()
    
    for series in iterator:
      times_set.intersection_update(series.TimesSet())

    return sorted(times_set)

  def _DefaultMapInputFunction(self, row):
    sorted_keys = sorted(row.keys())
    return [row[key] for key in sorted_keys]
