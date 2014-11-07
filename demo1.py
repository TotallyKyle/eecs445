# -*- coding: utf-8 -*-
""" 
Example of use multi-layer perceptron
=====================================

Task: Approximate USD_JPY exchnage rate

"""

import neurolab as nl
import numpy as np
import InputModels as model
import FeatureParser as parser

# Create train samples, in this case the input data model
# has 5 features, to predict the exchange rate on day N
# the feature is the exchange rate on day N-1 ... N-5
# data is a tuple where 
# data[0] = input, data[1] = output
# data[2] = min_val, data[3] = max_val in this sample model

data = model.timeDelayed(5)

# we still need to implement validation so 0% is used for validation
# 70% of the data is used for training and 30% for testing here
segmented_input = parser.segmentation(data[0], 0.7, 0, 0.3)
segmented_target = parser.segmentation(data[1], 0.7, 0, 0.3)
train_input = segmented_input[0]
train_target = segmented_target[0]

test_input = segmented_input[2]
test_target = segmented_target[2]

train_size = len(train_input)
test_size = len(test_input)
# we need to convert the format of the data to be
# compliant with the neurolab API, print out the
# values of inp and tar to see format
train_input = np.array(train_input)
train_target = np.array(train_target)
train_target = train_target.reshape(train_size, 1)


min_val = data[2]
max_val = data[3]

# Create network with 3 layers with 5, 5, and 1 neruon(s) in each layer 
# and randomly initialized
# the first list refers to the range of values for each input feature, 
# the length of the second list is the # of layers and each number is the # of nerons
# EX: net = nl.net.newff([[-0.5, 0.5], [-0.5, 0.5], [-1,10]], [5, 3, 1])
net = nl.net.newff([[min_val,max_val],[min_val,max_val],[min_val,max_val],[min_val,max_val],[min_val,max_val]],[5, 5, 1])

# Train network
error = net.train(train_input, train_target, epochs=500, show=100, goal=0.02)

# Simulate network on tet data
out = model.denormalize_target(net.sim(test_input).reshape(1,test_size).tolist()[0], min_val, max_val)

# Plot result
import pylab as pl
# pl.subplot(211)
# pl.plot(error)
# pl.xlabel('Epoch number')
# pl.ylabel('error (default SSE)')

# x2 = np.linspace(-6.0,6.0,150)
# y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)

# y3 = out.reshape(size)

# pl.subplot(212)
pl.plot(range(test_size), out, '-',range(test_size) , model.denormalize_target(test_target, min_val, max_val), '.')
pl.legend(['prediction value', 'actual value'])
pl.show()
