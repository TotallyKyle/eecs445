from  TimeSeriesDataFile  import  TimeSeriesDataFile
from  FeatureSetOptimizer import  FeatureSetOptimizer

f  = TimeSeriesDataFile('USDJPY_complete.csv', TimeSeriesDataFile.Formats.Standard)
f2 = TimeSeriesDataFile('initial_features_edited.csv', TimeSeriesDataFile.Formats.Alternate)

# Extract features
f.ExtractAllFeatures()
f2.ExtractAllFeatures()

# Get Extracted Features
f_features  = f.ExtractedFeatures()
f2_features = f2.ExtractedFeatures()

# Feature Set Optimizer
feature_set_optimizer = FeatureSetOptimizer()

# Define time series features
feature_set_optimizer.AddDataColumn(f_features['Close'])
feature_set_optimizer.AddDataColumn(f_features['Volume'])
feature_set_optimizer.AddDataColumn(f2_features['NYK'])
feature_set_optimizer.AddDataColumn(f2_features['DJIA USA'])

# Define possible feature key set
feature_set_optimizer.FeatureKeys = [
  'Close',
  'Close -1',
  'Close -2',
  'Close -3',
  'Close -4',
  'Close -5',
  'Close MA5',
  'Close MA30',
  'NYK',
  'DJIA USA',
  'Volume'
]

def row_filterer(feature_key_set, row_idx, num_rows, date):
  if 'Close MA30' in feature_key_set:
    return row_idx - 30 >= 0
  elif 'Close MA5'  in feature_key_set:
    return row_idx - 5 >= 0
  elif 'Close -5' in feature_key_set:
    return row_idx - 5 >= 0
  elif 'Close -4' in feature_key_set:
    return row_idx - 4 >= 0
  elif 'Close -3' in feature_key_set:
    return row_idx - 3 >= 0
  elif 'Close -2' in feature_key_set:
    return row_idx - 2 >= 0
  elif 'Close -1' in feature_key_set:
    return row_idx - 1 >= 0
  
  # due to output - don't want to keep last row
  return row_idx + 1 < num_rows

def row_feature_mapper(feature_key_set, row_idx, time_series_data):
  row_features = {}
  if 'Close' in feature_key_set:
    row_features['Close']    = time_series_data[row_idx]['Close']
  if 'Close -1' in feature_key_set:
    row_features['Close -1'] = time_series_data[row_idx - 1]['Close']
  if 'Close -2' in feature_key_set:
    row_features['Close -2'] = time_series_data[row_idx - 2]['Close']
  if 'Close -3' in feature_key_set:
    row_features['Close -3'] = time_series_data[row_idx - 3]['Close']
  if 'Close -4' in feature_key_set:
    row_features['Close -4'] = time_series_data[row_idx - 4]['Close']
  if 'Close -5' in feature_key_set:
    row_features['Close -5'] = time_series_data[row_idx - 5]['Close']
  if 'NYK' in feature_key_set:
    row_features['NYK']      = time_series_data[row_idx]['NYK']
  if 'DJIA USA' in feature_key_set:
    row_features['DJIA USA'] = time_series_data[row_idx]['DJIA USA']
  if 'Volume' in feature_key_set:
    row_features['Volume']   = time_series_data[row_idx]['Volume']
  
  # Ignore these for now
  if 'Close MA5' in feature_key_set:
    1 + 1
  if 'Close MA30' in feature_key_set:
    1 + 1
  return row_features

def row_output_mapper(row_idx, time_series_data):
  return {'Close +1' : time_series_data[row_idx + 1]['Close']}


# Define feature set filtering and mapping functions
feature_set_optimizer.RowFilterer       = row_filterer;
feature_set_optimizer.RowFeatureMapper  = row_feature_mapper;
feature_set_optimizer.RowOutputMapper   = row_output_mapper;

# Run the actual model
feature_set_optimizer.OptimizeFeatureSet()
