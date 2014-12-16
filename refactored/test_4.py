from   TimeSeriesDataFile import  TimeSeriesDataFile
from   ModelDataBuilder   import  ModelDataBuilder
from   ErrorAnalyser      import  ErrorAnalyser
from   sklearn            import  cross_validation
import numpy              as      np
import neurolab           as      nl
from   AccuracyEvaluation import  RMSE
from   Queue              import  PriorityQueue
from   neurolab_extension import  net_init
import json


# Initialize references to files
f  = TimeSeriesDataFile('USDJPY_complete.csv', TimeSeriesDataFile.Formats.Standard)
f2 = TimeSeriesDataFile('initial_features_edited.csv', TimeSeriesDataFile.Formats.Alternate)

initial_values = [{'b': np.array([ -0.62295578,   1.84040558,  -1.17521869,   0.46117341,
        -0.69931897,   0.27616017, -14.07748004,   0.58938518,
        -0.69103759,   1.04569684,   1.76566579]), 'w': np.array([[ -1.07903589e+00,   4.97671758e+00,   1.95776391e+00,
          4.87427642e-01,  -1.59896773e+00,  -5.52375810e+00,
          6.92004816e-04,   1.03967275e-01,   9.20412905e-01,
         -1.43968887e-03,   1.15626445e-02],
       [ -3.75658523e+00,   1.94021712e+00,   4.73598985e-01,
         -2.96529026e+00,  -3.64919148e-01,  -3.79376276e+00,
         -3.38924062e-03,   1.63711319e+00,  -9.20406560e-01,
          5.39143593e-02,  -2.82210270e-04],
       [ -8.09946785e-01,  -7.45527655e-01,  -7.46886244e-01,
         -7.03254704e-01,  -7.07240943e-01,  -9.08304341e-01,
          1.69330331e+00,  -8.95616908e-01,  -7.10710260e-01,
          1.89377526e+00,  -1.29757020e-01],
       [ -3.22134762e-02,  -9.69238626e-03,  -1.53014945e-02,
          1.60141646e-02,   1.04602239e-02,   1.15033898e-02,
         -4.62023939e-01,   2.65886110e-02,   2.50615798e-02,
         -3.54325457e-01,  -2.56394094e-02],
       [  6.37375202e-03,   8.33442678e-03,   3.12939636e-02,
          2.52494733e-02,  -2.72614319e-02,   3.06674186e-02,
          3.75125713e-02,  -5.99217222e-03,   1.70070183e-02,
          1.23507114e-02,  -5.32778697e-05],
       [  1.01390410e+00,  -3.14227413e-01,   5.26864489e-01,
          1.38411353e-01,  -1.84781577e+00,   9.48221116e-01,
         -1.17446839e-01,   2.74078437e-01,  -7.55333138e-02,
          2.25382878e-01,  -1.62885904e-02],
       [  7.31009498e-03,  -6.73142748e-03,   2.55968048e-03,
         -1.48810386e-02,   1.56213741e-03,   1.09309034e-01,
         -7.68512790e-06,  -5.46505115e-03,   6.97218543e-03,
          1.55281444e-05,  -8.32930716e-09],
       [ -2.12317853e+00,  -1.45945329e+00,   8.68509531e-01,
          2.43598401e+00,  -9.97004676e-01,  -2.33860715e+00,
          2.74459762e-02,   2.60736683e+00,  -2.64540504e-01,
          1.03683402e-01,  -9.45555079e-03],
       [ -6.85235072e-02,   5.16634840e-02,   2.09201397e-02,
         -2.32516648e-02,   2.66758798e-02,  -5.83308176e-02,
          6.69872922e-03,   4.87993328e-02,  -4.28657717e-05,
         -2.08636616e-02,  -1.23758699e-05],
       [  2.88993875e-02,   2.60085176e-02,   5.92262408e-03,
          2.29261316e-02,   3.22360898e-02,   2.05319587e-02,
          5.08970142e-01,  -9.64985643e-03,  -2.01939199e-02,
          3.09230786e-01,  -1.35182262e-03],
       [  2.07870964e-03,  -4.46700739e-05,   3.06894297e-03,
          6.25198139e-04,   8.20066492e-04,  -1.72920340e-02,
          1.67833281e-07,   1.04405923e-04,  -7.04925125e-03,
          1.98736328e-07,   2.84457188e-10]])}, {'b': np.array([ 2.67770532, -1.12459363, -2.27514869,  0.74613734,  2.28848334]), 'w': np.array([[ 0.92810016, -1.05712782,  4.79442243, -1.38164365,  1.05176991,
         4.09507764, -0.6147755 , -4.13945581, -1.05846473,  1.06415603,
         0.40923276],
       [ 1.63719177, -0.10314372, -0.00932966, -1.60899183, -0.37590867,
         0.02335719,  2.90422743, -0.02760121,  0.25590918, -0.29949032,
        -1.34895727],
       [-3.18625163,  2.27506711, -1.00276806,  0.11077076, -2.27716572,
        -0.13907818, -3.93283977, -0.47603931,  2.27518359, -2.27766589,
         1.14133373],
       [ 0.41348323,  0.14129358,  0.11477431, -0.3399907 , -0.07194391,
        -3.15807324,  0.07193338,  3.21191048,  0.07195688, -0.08667112,
         0.38603113],
       [-1.86669501,  0.70690159, -0.02350768, -1.23182632,  0.65294038,
         0.01823029,  1.48362618, -0.02134072, -0.87704261,  0.70584622,
        -1.15288426]])}, {'b': np.array([ 2.32495693]), 'w': np.array([[ 5.80240518,  6.10279995,  4.82840441,  0.35600683,  2.79337199]])}]

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
net_init(net, initial_values)

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

print RMSE([data[0] for data in predicted], [data[0] for data in actual])
