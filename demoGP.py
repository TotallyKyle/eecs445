# -*- coding: utf-8 -*-
""" 
Example of GP class used to predict USD_JPY exchange rate
=====================================

"""


import numpy as np
import InputModels as model
import FeatureParser as parser
import random
from scipy.cluster.vq import kmeans,vq

# Create train samples, in this case the input data model
# has 5 features, to predict the exchange rate on day N
# the feature is the exchange rate on day N-1 ... N-5
# data is a tuple where 
# data[0] = input, data[1] = output
# data[2] = min_val, data[3] = max_val in this sample model
data = []
feature_value_range = {}
joined_data = []
joined_data = parser.add_feature_to_data_alt(data, 'initial_features_edited.csv', 'DJIA USA')
data = joined_data[0]
feature_value_range['DJIA USA'] = joined_data[1]
joined_data = parser.add_feature_to_data_alt(data, 'initial_features_edited.csv', 'NYK')
data = joined_data[0]
feature_value_range['NYK'] = joined_data[1]
# joined_data = parser.add_feature_to_data(data, 'USDJPY_complete.csv', 'High')
# data = joined_data[0]
# feature_value_range['High'] = joined_data[1]
# joined_data = parser.add_feature_to_data(data, 'USDJPY_complete.csv', 'Volume')
# data = joined_data[0]
# feature_value_range['Volume'] = joined_data[1]


#add time series features
timeSeries = model.timeDelayedFeature(5, 'USDJPY_complete.csv')
timeSeriesFeature = timeSeries[0]
feature_value_range['0'] = timeSeries[1]
feature_value_range['1'] = timeSeries[1]
feature_value_range['2'] = timeSeries[1]
feature_value_range['3'] = timeSeries[1]
feature_value_range['4'] = timeSeries[1]

data = parser.join_on_minimum(data, timeSeriesFeature)



feature_value_range = parser.convert_feature_value_range(feature_value_range);
data = parser.convert_input(data)

minVal = min(target)
maxVal = max(target)
# we still need to implement validation so 0% is used for validation
# 70% of the data is used for training and 30% for testing here
segmented_input = parser.segmentation(data, 0.7, 0, 0.3)

train_input = segmented_input[0]

test_input = segmented_input[2]


#COMBINE TRAIN INPUT AND TARGET
#hybrid_train = train_input;
#for i in range(len(train_input)):
#	hybrid_train[i].append(train_target[i])
	
train_size = len(train_input)
test_size = len(test_input)
# we need to convert the format of the data to be
# compliant with the neurolab API, print out the
# values of inp and tar to see format
train_input = np.array(train_input)
