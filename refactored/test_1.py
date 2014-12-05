from   TimeSeriesDataFile import  TimeSeriesDataFile
from   ModelDataBuilder   import  ModelDataBuilder
import numpy              as      np
import neurolab           as      nl

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
data_builder.AddFeatureTimeSeries(f2_features['DJIA USA'])
data_builder.AddFeatureTimeSeries(f2_features['NYK'])
data_builder.BuildRawTimeSeries()

def filter_func (row, idx, raw_data, date):
  return idx - 5 > 0 and idx + 1 < len(raw_data)

def map_input_func (row, idx, raw_data, date):
  prev_times = {}
  for i in range(1, 6):
    prev_times[str(i)] = raw_data[idx - i]['Close']

  row.update(prev_times)
  return row

# Give time series value
def map_target_func (row, idx, raw_data, date):
  return [row['Close']]

[inputs, targets, feature_ranges] = data_builder.MapToArrays(map_target_func, filter_func, map_input_func)

net = nl.net.newff(feature_ranges, [7, 5, 1])
nl.init.midpoint(net.layers[1])

inputs = np.array(inputs)
targets = np.array(ModelDataBuilder.Normalize(targets))

error = net.train(
  inputs,
  targets,
  epochs=500,
  show=100,
  goal=0.02
)

print 'Error : ' + str(error)
