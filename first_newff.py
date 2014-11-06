# -*- coding: utf-8 -*-
""" 
Example of use multi-layer perceptron
=====================================

Task: Approximation function: 1/2 * sin(x)

"""

import neurolab as nl
import numpy as np
import InputModels as model

# Create train samples
x = np.linspace(-7, 7, 20)
y = np.sin(x) * 0.5

#size = len(x)
#inp = x.reshape(size,1)
#tar = y.reshape(size,1)

data = InputModels.timeDelayed(5)
size = len(data[0])
inp = np.array(data[0])
tar = np.array(data[1])
tar = tar.reshape(size, 1)
#orig = 
#inp = x.reshape(size,1)
#tar = y.reshape(size,1)

min_val = data[2]
max_val = data[3]
# Create network with 2 layers and random initialized
#the first list refers to the range values for each input, 
#the length of the second list is the # of layers and each number is the # of nerons
#net = nl.net.newff([[-0.5, 0.5], [-0.5, 0.5], [-1,10]], [5, 3, 1])
net = nl.net.newff([[min_val,max_val],[min_val,max_val],[min_val,max_val],[min_val,max_val],[min_val,max_val]],[5, 5, 1])

# Train network
error = net.train(inp, tar, epochs=500, show=100, goal=0.02)

# Simulate network
out = InputModels.denormalize_target(net.sim(inp).reshape(1,size).tolist()[0], min_val, max_val)
