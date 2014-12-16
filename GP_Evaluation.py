import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import AccuracyEvaluation as ac
import pylab as P
# put this back in GP
def evaluate_by_feature(results, targets, features, feature):
	num_data = len(results)
	num_days = len(results[0])
	feature = features.index(feature)
	rmse = {}
	for day in range(1, num_days):
		prediction = [data_pt[day][0][feature] for data_pt in results]
		target = [row[feature for row in targets[1:num_data+1]]]
		rmse[str(day)] = ac.RMSE(prediction, target)

	# graph info using MSE with bar graph


def evaluate_by_feature_with_distribution
	# mu, sigma = 200, 25
	# x = mu + sigma*P.randn(10000)

	# # the histogram of the data with histtype='step'
	# n, bins, patches = P.hist(x, 100, normed=1, histtype='stepfilled')
	# P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

	num_data = len(results)
	num_days = len(results[0])
	feature = features.index(feature)
	error = {}
	for day in range(1, num_days):
		prediction = [data_pt[day][0][feature] for data_pt in results]
		target = [row[feature for row in targets[1:num_data+1]]]
		error[str(day)] = ac.percentage_diff(prediction, target)
		n, bins, patches = P.hist(error[str(day)], 25, normed=1, histtype='stepfilled')
		P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
		P.figure()
	P.show()