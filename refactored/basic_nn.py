from   TimeSeriesDataFile import  TimeSeriesDataFile
from   ModelDataBuilder   import  ModelDataBuilder
from   AccuracyEvaluation import  RMSE
import math
import numpy              as      np
import neurolab           as      nl
from   neurolab_extension import  net_init

def Filter(idx, num_rows, date):
  return idx - 30 >= 0 and idx + 1 < num_rows

def MapInput(idx, data):
  features = TimeSeriesFeature.Generate(idx, data, 'Close', 5)
  features['MA5']  = MovingAverageFeature.Generate(idx, data, 'Close', 5)
  features['MA10'] = MovingAverageFeature.Generate(idx, data, 'Close', 10)
  features['MA30'] = MovingAverageFeature.Generate(idx, data, 'Close', 30)
  features['Log Returns'] = LogReturnsFeature.Generate(idx, data, 'Close', 5)
  features['Log Returns'] = LogReturnsFeature.Generate(idx, data, 'Close', 15)

  return features

def MapOutput(idx, data):
  return {
    'Close' : data[idx + 1]['Close']
  }

class Trendifier:
  @staticmethod
  def TredifyArray(arr, days_back = 1):
    trends  = []
    old_val = arr[0]
    for i in range(1, len(arr)):
      new_val = arr[i]
      if old_val < new_val:
        trends.append(1)
      elif old_val == new_val:
        trends.append(0)
      else:
        trends.append(-1)
    return trends

class TimeSeriesFeature:
  @staticmethod
  def Generate(idx, data, feature, days_back):
    features = {}
    for i in range(0, days_back):
      features[feature + ' -' + str(i)] = data[idx - i][feature]
    return features

class MovingAverageFeature:
  @staticmethod
  def Generate(idx, data, feature, days_back):
    return sum([row[feature] for row in data[idx-days_back:idx]]) / days_back

class LogReturnsFeature:
  @staticmethod
  def Generate(idx, data, feature, days_back):
    return math.log(data[idx][feature]/data[idx - days_back][feature])

def single_run(inputs, targets, ranges, r):
  # Initialize NN
  net = nl.net.newff(ranges, [len(inputs[0]), 5, len(targets[0])])
  for layer in net.layers:
    nl.init.initwb_reg(layer)

  # net_init(net)

  # Partition
  inputs_train  = inputs[:len(inputs) - 20]
  targets_train = targets[:len(targets) - 20]
  inputs_test   = inputs[-19:]
  targets_test  = targets[-19:]

  print 'Training'

  # Train NN
  error = net.train(
    inputs_train,
    targets_train,
    epochs=500,
    show=100,
    goal=0.02
  )

  print 'Trained'

  print [row[0] for row in targets_test]

  print 'Simulating'

  actual    = [row[0] for row in ModelDataBuilder.Denormalize(targets_test, r)]
  predicted = [row[0] for row in ModelDataBuilder.Denormalize(net.sim(inputs_test), r)]

  print 'Simulated'

  for i in range(len(actual)):
    print actual[i], predicted[i]
  
  rmse = RMSE(predicted, actual)

  print rmse

  trend_actual    = Trendifier.TredifyArray(actual)
  trend_predicted = Trendifier.TredifyArray(predicted)
  
  correct = 0  
  for i in range(len(trend_actual)):
    print trend_actual[i], trend_predicted[i]
    if trend_predicted[i] == trend_predicted[i]:
      correct += 1

  accuracy = float(correct)/float(len(trend_actual))

  print accuracy

  print "Num Correct : " + str(correct)
  print "Accuracy    : " + str(accuracy)

  return accuracy, rmse, [layer.np for layer in net.layers]

def record(accuracy, rmse, nn_init_vals):
  f = file('optimal_init/tmp1-Close-MA-LgRet.txt', 'w')
  f.write(str(accuracy) + '\n\n')
  f.write(str(rmse) + '\n\n')
  f.write(str(nn_init_vals) + '\n\n')

def runner():

  f  = TimeSeriesDataFile('../USDJPY_complete.csv', TimeSeriesDataFile.Formats.Standard)
  f2 = TimeSeriesDataFile('../initial_features_edited.csv', TimeSeriesDataFile.Formats.Alternate)

  f.ExtractAllFeatures()
  f2.ExtractAllFeatures()

  # Get Extracted Features
  f_features  = f.ExtractedFeatures()
  f2_features = f2.ExtractedFeatures()

  data_builder = ModelDataBuilder()
  data_builder.AddFeatureTimeSeries(f_features['Close'])

  data_builder.BuildRawTimeSeries()
  inputs, targets, ranges = data_builder.MapToArrays(Filter, MapInput, MapOutput)

  inputs = np.array(inputs)
  targets, r = np.array(ModelDataBuilder.Normalize(targets))

  best_accuracy = 0
  best_rmse     = 1000
  i = 0
  while True:
    i += 1
    print 'Iteration : ', i, ' Best accuracy : ', best_accuracy, ' Best RMSE : ', best_rmse
    accuracy, rmse, nn_init_vals = single_run(inputs, targets, ranges, r)
    if accuracy > best_accuracy:
      record(accuracy, rmse, nn_init_vals)
      best_accuracy = accuracy
      best_rmse     = rmse


# Run demo
runner()


