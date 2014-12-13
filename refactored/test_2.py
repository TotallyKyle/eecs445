from  TimeSeriesDataFile  import  TimeSeriesDataFile
from  FeatureSetOptimizer import  FeatureSetOptimizer
from  GP                  import  GP
from  AccuracyEvaluation  import  *

f  = TimeSeriesDataFile('USDJPY_complete.csv', TimeSeriesDataFile.Formats.Standard)
f2 = TimeSeriesDataFile('initial_features_edited.csv', TimeSeriesDataFile.Formats.Alternate)

# Extract features
f.ExtractAllFeatures()
f2.ExtractAllFeatures()

# Get Extracted Features
f_features  = f.ExtractedFeatures()
f2_features = f2.ExtractedFeatures()

# Feature Set Optimizer
feature_set_optimizer = FeatureSetOptimizer(5)

# Define time series features
feature_set_optimizer.AddDataColumn(f_features['Close'])
feature_set_optimizer.AddDataColumn(f_features['Volume'])
feature_set_optimizer.AddDataColumn(f2_features['NYK'])
feature_set_optimizer.AddDataColumn(f2_features['DJIA USA'])
feature_set_optimizer.AddDataColumn(f2_features['S&P/TSX Composite (Canadian Index)'])
feature_set_optimizer.AddDataColumn(f2_features['(UK Index) UKX Index'])
feature_set_optimizer.AddDataColumn(f2_features['(HengSeng Index) HSI Index'])
feature_set_optimizer.AddDataColumn(f_features['Open'])
feature_set_optimizer.AddDataColumn(f_features['High'])
feature_set_optimizer.AddDataColumn(f_features['Low'])

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
  'S&P/TSX Composite (Canadian Index)',
  '(UK Index) UKX Index',
  '(HengSeng Index) HSI Index',
  'Open',
  'High',
  'Low'
]

feature_set_optimizer.RequiredKeys = [
  'Close'
]

def row_filterer(feature_key_set, row_idx, num_rows, date):
  if row_idx + 2 > num_rows:
    return False
  elif 'Close MA30' in feature_key_set:
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
  return True

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
  if 'High' in feature_key_set:
    row_features['High']   = time_series_data[row_idx]['High']
  if 'Low' in feature_key_set:
    row_features['Low']   = time_series_data[row_idx]['Low']
  if 'Open' in feature_key_set:
    row_features['Open']   = time_series_data[row_idx]['Open']
  if 'S&P/TSX Composite (Canadian Index)' in feature_key_set:
    row_features['S&P/TSX Composite (Canadian Index)'] = time_series_data[row_idx]['S&P/TSX Composite (Canadian Index)']
  if '(UK Index) UKX Index' in feature_key_set:
    row_features['(UK Index) UKX Index'] = time_series_data[row_idx]['(UK Index) UKX Index']
  if '(HengSeng Index) HSI Index' in feature_key_set:
    row_features['(HengSeng Index) HSI Index'] = time_series_data[row_idx]['(HengSeng Index) HSI Index']
  if 'Close MA5' in feature_key_set:
    row_features['Close MA5'] = sum([point['Close'] for point in time_series_data[row_idx-5:row_idx]]) / 5
  if 'Close MA30' in feature_key_set:
    row_features['Close MA30'] = sum([point['Close'] for point in time_series_data[row_idx-30:row_idx]]) / 30
  return row_features

def row_output_mapper(row_idx, time_series_data):
  return {'Close +1' : time_series_data[row_idx + 1]['Close']}


# Define feature set filtering and mapping functions
feature_set_optimizer.RowFilterer       = row_filterer;
feature_set_optimizer.RowFeatureMapper  = row_feature_mapper;
feature_set_optimizer.RowOutputMapper   = row_output_mapper;

# Returns some kind of error and any other data that should be stored
def gp_model (training_data, output_data):

  # train the model
  gp = GP(training_data)
  
  # simulate the model
  output = gp.predict_feature('Close')[0]
  target = gp.create_target('Close')[1]
  
  # we return some kind of error
  return RMSE(output, target), None

# Run the actual model
print feature_set_optimizer.OptimizeFeatureSet(gp_model)
