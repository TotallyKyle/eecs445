from   TimeSeriesDataFile import  TimeSeriesDataFile
from   ModelDataBuilder   import  ModelDataBuilder
from   ErrorAnalyser      import  ErrorAnalyser
from   sklearn            import  cross_validation
import matplotlib.pyplot  as      plt
import numpy              as      np
import neurolab           as      nl
from   GP                 import  GP
import GP_Evaluation      as evaluate




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
data_builder.AddFeatureTimeSeries(f_features['RSI'])
data_builder.AddFeatureTimeSeries(f2_features['DJIA USA'])
data_builder.AddFeatureTimeSeries(f2_features['NYK'])
data_builder.BuildRawTimeSeries()

def filter_func (idx, num_rows, date):
  return idx - 30 >= 0 and idx + 2 < num_rows

def output_filter_func (row, idx, raw_data, date):
  return idx - 30 >= 0 and idx + 2 < len(raw_data)

def map_input_func (idx, raw_data):
  row = raw_data[idx]
  prev_times = {}
  moving_avgs = {}
  for i in range(1, 6):
    prev_times[str(i)] = raw_data[idx - i]['Close']
  row.update(prev_times)
  data = raw_data[idx-5:idx]
  moving_avgs['MA5'] = sum([point['Close'] for point in data]) / 5
  data = raw_data[idx-30:idx]
  moving_avgs['MA30'] = sum([point['Close'] for point in data]) / 30
  data = raw_data[idx-1]['Volume']*(raw_data[idx-1]['Close'] - raw_data[idx-2]['Close'])/raw_data[idx-2]['Close']
  moving_avgs['Trend'] = data
  row.update(moving_avgs)
  return row

# Give time series value
def map_target_func (idx, raw_data):
  outputs = {
    'Close'  : raw_data[idx + 1]['Close'],
    'Close Tomorrow' : raw_data[idx + 2]['Close']
  }

  return outputs

data, _ = data_builder.GenerateFeatures(filter_func, map_input_func, map_target_func)

for row in data:
  row.pop('Volume', None)
gp = GP(data)
output = gp.predict(4, gp.test_data[0:100])
evaluate.evaluate_by_feature_with_distribution(output[0], gp.test_data, gp.features, 'Close')
evaluate.line_graph(output, gp.test_data, gp.features, 'Close')

num_data = len(output[0])
num_days = len(output[0][0])
feature = gp.features.index('Close')
mu = output[0]
s = output[1]
for day in range(1, num_days):
  predictions = []
  targets = []
  prediction = np.array([data_pt[day][0][feature] for data_pt in mu])
  target = np.array([row[feature] for row in gp.test_data[day:num_data+day]])
  predictions = np.concatenate([prediction, predictions])
  targets = np.concatenate([target, targets])

  trend_output = predictions[1:]
  trend_test = targets[1:]
  flag = False
  counter = 0
  for x in predictions:
    if(flag):
      if(x >= prev):
        trend_output[counter] = 1
      else:
        trend_output[counter] = -1
      counter += 1
    flag = True
    prev = x

  counter = 0
  flag = False
  for x in targets:
    if(flag):
      if(x >= prev):
        trend_test[counter] = 1
      else:
        trend_test[counter] = -1
      counter += 1
    flag = True
    prev = x

  x = np.arange(min([len(trend_test), len(trend_output)]))
  counter = 0.0
  for y in x:
    if(trend_output[y] == trend_test[y]):
      counter +=1.0

  print counter / min([len(trend_test), len(trend_output)])
  plt.plot(x, trend_output[x])  
  plt.plot(x, trend_test[x])

  plt.legend(['Predicted', 'Actual'], loc='upper right')
  plt.axis([0, len(trend_output), -3, 3])
  plt.show()