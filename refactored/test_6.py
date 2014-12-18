
def filter_func(idx, num_rows, date):
  return idx + 1 < num_rows and idx - 30 > 0

def moving_avg(idx, data, i):
  return sum([point['Close'] for point in data[idx-i:idx]]) / i

def map_input_func(idx, data):
  feature_data = {}
  
  for i in range(0, 100):
    feature_data['Close -' + str(i)] = data[idx - i]['Close']

  for i in range(0, 5):
    feature_data['NYK - ' + str(i)] = data[idx - i]['NYK']

  for i in range(0, 5):
    feature_data['DJIA - ' + str(i)] = data[idx - i]['DJIA']
  
  # Moving averages
  for i in range(1, 4):
    feature_data['Close MA - ', i * 10] = moving_avg(idx, data, i * 10)

  return feature_data


def map_target_func(idx, data):
  return {
    'Close +1' : data[idx + 1]['Close']
  }


# Initialize references to files
f  = TimeSeriesDataFile('USDJPY_complete.csv', TimeSeriesDataFile.Formats.Standard)
f2 = TimeSeriesDataFile('initial_features_edited.csv', TimeSeriesDataFile.Formats.Alternate)

# Extract features
f.ExtractAllFeatures()
f_features.ExtractedFeatures()

f2.ExtractAllFeatures()
f2_features.ExtractedFeatures()

# Build Features and Targets 
builder = ModelDataBuilder()
data_builder.AddFeatureTimeSeries(f_features['Close'])
data_builder.AddFeatureTimeSeries(f2_features['DJIA USA'])
data_builder.AddFeatureTimeSeries(f2_features['NYK'])

# Generate inputs and targets arrays
inputs, targets, feature_ranges = data_builder.MapToArrays(map_target_func, filter_func, map_input_func)
