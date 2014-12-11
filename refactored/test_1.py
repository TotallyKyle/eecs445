from   TimeSeriesDataFile import  TimeSeriesDataFile
from   ModelDataBuilder   import  ModelDataBuilder
from   ErrorAnalyser      import  ErrorAnalyser
from   sklearn            import  cross_validation
import numpy              as      np
import neurolab           as      nl

while True:

  # Initialize references to files
  f  = TimeSeriesDataFile('USDJPY_complete.csv', TimeSeriesDataFile.Formats.Standard)
  f2 = TimeSeriesDataFile('initial_features_edited.csv', TimeSeriesDataFile.Formats.Alternate)

  # Extract features
  f.ExtractAllFeatures()
  f2.ExtractAllFeatures()

  # Get Extracted Features
  f_features  = f.ExtractedFeatures()
  f2_features = f2.ExtractedFeatures()

  # Need to Intersect on dates
  data_builder = ModelDataBuilder(train = .7, test = .3)
  data_builder.AddFeatureTimeSeries(f_features['Close'])
  data_builder.AddFeatureTimeSeries(f_features['Volume'])
  data_builder.AddFeatureTimeSeries(f2_features['DJIA USA'])
  data_builder.AddFeatureTimeSeries(f2_features['NYK'])
  data_builder.BuildRawTimeSeries()

  def input_filter_func (row, idx, raw_data, date):
    return idx - 30 >= 0 and idx + 2 < len(raw_data)

  def output_filter_func (row, idx, raw_data, date):
    return idx - 30 >= 0 and idx + 2 < len(raw_data)

  def map_input_func (row, idx, raw_data, date):
    prev_times = {}
    moving_avgs = {}
    for i in range(1, 6):
      prev_times[str(i)] = raw_data[idx - i]['Close']
    row.update(prev_times)
    data = raw_data[idx-5:idx]
    moving_avgs['MA5'] = sum([point['Close'] for point in data]) / 5
    data = raw_data[idx-30:idx]
    moving_avgs['MA30'] = sum([point['Close'] for point in data]) / 30
    row.update(moving_avgs)
    return row

  # Give time series value
  def map_target_func (row, idx, raw_data, date):
    outputs = {
      'Close'  : raw_data[idx + 1]['Close'],
      'Close Tomorrow' : raw_data[idx + 2]['Close']
    }

    return outputs

  data_builder.AugmentRawData(input_filter_func, map_input_func)
  data = data_builder.GetRawData()

  inputs = data_builder.MapInputToArrays()
  targets = data_builder.MapOutputToArrays(map_target_func, output_filter_func)
  feature_ranges = data_builder.MapRangesToArray()

  # Initialize some nn
  net = nl.net.newff(feature_ranges, [len(inputs[0]), 5, len(targets[0])])
  nl.init.midpoint(net.layers[1])

  inputs = np.array(inputs)
  targets, r = np.array(ModelDataBuilder.Normalize(targets))

  # Split for cross validation
  inputs_train, inputs_test, targets_train, targets_test = \
    cross_validation.train_test_split(inputs, targets, test_size=0.2, random_state=0)

  error = net.train(
    inputs_train,
    targets_train,
    epochs=500,
    show=100,
    goal=0.02
  )


  predicted = ModelDataBuilder.Denormalize(net.sim(inputs_test), r)
  actual    = ModelDataBuilder.Denormalize(targets_test, r)

  labeled_predicted = data_builder.MapBackToOutputs(predicted)
  labeled_actual    = data_builder.MapBackToOutputs(actual)

  print labeled_predicted[-5:]
  print labeled_actual[-5:]

  analyser = ErrorAnalyser(labeled_predicted, labeled_actual)
  print analyser.AverageErrorForOutput('Close')
  print analyser.AverageErrorForOutput('Close Tomorrow')
  # v.GenerateOutputErrorHistogram('Close')
  # v.GenerateLinearErrorPlot(['Close, Next Day Close'])
