# -*- coding: utf-8 -*-
""" 
Example of use multi-layer perceptron
=====================================

Task: Approximation function: 1/2 * sin(x)

"""

import neurolab as nl
import numpy as np
import InputModels as model
import FeatureParser as parser
import random
from scikits.learn.gmm import GMM

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

# construct the target value vector
target = []
target_val = parser.add_feature_to_data(target, 'USDJPY_complete.csv', 'Close')
target = parser.match_target_to_data(target_val[0], data)


# this converts the data structure from list of dicts to list of lists
# for both input and target
target = parser.convert_input(target)
target = [val for sublist in target for val in sublist]
target = model.normalize_target(target, target_val[1][0], target_val[1][1])

feature_value_range = parser.convert_feature_value_range(feature_value_range);
data = parser.convert_input(data)

minVal = min(target)
maxVal = max(target)
# we still need to implement validation so 0% is used for validation
# 70% of the data is used for training and 30% for testing here
segmented_input = parser.segmentation(data, 0.7, 0, 0.3)
segmented_target = parser.segmentation(target, 0.7, 0, 0.3)
train_input = segmented_input[0]
train_target = segmented_target[0]

test_input = segmented_input[2]
test_target = segmented_target[2]
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
train_target = np.array(train_target)
train_target = train_target.reshape(train_size, 1)

g = GMM(n_states=2, n_dim=len(train_input[0])