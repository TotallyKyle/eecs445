from   TimeSeriesDataFile import  TimeSeriesDataFile
from   ModelDataBuilder   import  ModelDataBuilder
from   ErrorAnalyser      import  ErrorAnalyser
from   sklearn            import  cross_validation
import numpy              as      np
import neurolab           as      nl
from   AccuracyEvaluation import  RMSE
from   Queue              import  PriorityQueue
import json

best_found = [
  (1000, None),
  (1000, None),
  (1000, None),
  (1000, None),
  (1000, None),
  (1000, None),
  (1000, None),
  (1000, None),
  (1000, None),
  (1000, None)
];


while True:
  print 'Currently best found RMSE : ' + str(best_found[0][0])


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

  def input_filter_func (idx, num_rows, date):
    return idx - 30 >= 0 and idx + 2 < num_rows

  def output_filter_func (idx, num_rows, date):
    return idx - 30 >= 0 and idx + 2 < num_rows

  def map_input_func (idx, raw_data):
    prev_times = {}
    moving_avgs = {}
    row = raw_data[idx]
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
  def map_target_func (idx, raw_data):
    outputs = {
      'Close'  : raw_data[idx + 1]['Close']
    }

    return outputs

  inputs, targets, feature_ranges = data_builder.MapToArrays(map_target_func, input_filter_func, map_input_func)

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

  for i, (rmse, init_data) in enumerate(best_found):
    cur_rmse = RMSE([data[0] for data in predicted], [data[0] for data in actual])
    if cur_rmse < rmse:
      best_found[i] = cur_rmse, [layer.np for layer in net.layers]
      f = open('tmp/best_found.txt', 'w')
      f.write(str(best_found))
      break
