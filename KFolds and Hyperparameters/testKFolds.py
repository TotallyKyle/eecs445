from   TimeSeriesDataFile import  TimeSeriesDataFile
from   ModelDataBuilder   import  ModelDataBuilder
from   sklearn            import  cross_validation
import numpy              as      np
import neurolab           as      nl
import KFoldsValidation   as      k_folds


# Initialize references to files
f = TimeSeriesDataFile('USDJPY_complete.csv', TimeSeriesDataFile.Formats.Standard)
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
  return idx - 5 > 0 and idx + 2 < len(raw_data)

def map_input_func (row, idx, raw_data, date):
  prev_times = {}
  for i in range(1, 6):
    prev_times[str(i)] = raw_data[idx - i]['Close']

  row.update(prev_times)
  return row

# Give time series value
def map_target_func (row, idx, raw_data, date):
  outputs = {
    'Close'  : raw_data[idx + 1]['Close'],
    'Close Tomorrow' : raw_data[idx + 2]['Close']
  }

  return outputs

[inputs, targets, feature_ranges] = data_builder.MapToArrays(map_target_func, filter_func, map_input_func)

neuron_layer_size_list = [len(inputs[0]), 5, len(targets[0])] 
training_func = nl.train.train_gdm

error = k_folds.k_folds_validation(inputs, targets, feature_ranges, data_builder, neuron_layer_size_list, 5, training_func)
print error
