from   TimeSeriesDataFile import  TimeSeriesDataFile
from   ModelDataBuilder   import  ModelDataBuilder
from   ErrorAnalyser      import  ErrorAnalyser
from   sklearn            import  cross_validation
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
  row.pop('Volume')

inv_beta = [0.00001, 0.001, 0.01, 0.1, 1, 100]
inv_sigma = [0.001, 0.01, 0.1, 1, 10, 100]

gp = GP(data)
errors = evaluate.kfolds(data[:1501], inv_beta, inv_sigma)
errors = np.array(errors)
ideal = np.unravel_index(errors.argmin(), errors.shape)
print "best inverse beta:", inv_beta[ideal[0]]
print "best inverse sigma:", inv_sigma[ideal[1]]
