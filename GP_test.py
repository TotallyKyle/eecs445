
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import numpy as np
import InputModels as model
import FeatureParser as parser

""" This is code for simple GP regression. It assumes a zero mean GP Prior """


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


target = parser.convert_input(target)
target = [val for sublist in target for val in sublist]

data = parser.convert_input(data)

segmented_input = parser.segmentation(data, 0.7, 0, 0.3)
segmented_target = parser.segmentation(target, 0.7, 0, 0.3)
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

test_input = np.array(test_input)

# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x).flatten()
#f = lambda x: (0.25*(x**2)).flatten()


# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)


K = kernel(train_input, train_input)
L = np.linalg.cholesky(K)
Lk = np.linalg.solve(L, kernel(train_input, test_input))
mu = np.dot(Lk.T, np.linalg.solve(L, train_target))

N = 10         # number of training points.
n = 50         # number of test points.
s = 0.00005    # noise variance.

# Sample some input points and noisy versions of the function evaluated at
# these points. 
X = np.random.uniform(-5, 5, size=(N,1))
y = f(X) + s*np.random.randn(N)

K = kernel(X, X)
L = np.linalg.cholesky(K + s*np.eye(N))

# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# compute the mean at our test points.
Lk = np.linalg.solve(L, kernel(X, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, y))

# compute the variance at our test points.
K_ = kernel(Xtest, Xtest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)


# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-5, 5, -3, 3])

# draw samples from the prior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,10)))
pl.figure(2)
pl.clf()
pl.plot(Xtest, f_prior)
pl.title('Ten samples from the GP prior')
pl.axis([-5, 5, -3, 3])
pl.savefig('prior.png', bbox_inches='tight')

# draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
pl.figure(3)
pl.clf()
pl.plot(Xtest, f_post)
pl.title('Ten samples from the GP posterior')
pl.axis([-5, 5, -3, 3])
pl.savefig('post.png', bbox_inches='tight')

pl.show()
