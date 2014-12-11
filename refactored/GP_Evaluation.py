import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import AccuracyEvaluation as ac
import pylab as P
import GP as GP
# put this back in GP

def kfolds(train_data, train_target, inverse_beta, inverse_sigma, k=5):
	errors = [[0 for x in range(len(inverse_sigma))] for x in range(len(inverse_beta))] 
	data_size = len(train_data)
	segment_size = data_size / k
	partitioned_data = [train_data[i:i + segment_size] for i in range(0, len(train_data), segment_size)]
	partitioned_target = [train_target[i:i + segment_size] for i in range(0, len(train_target), segment_size)]
	for i in range(len(inverse_beta)):
		for j in range(len(inverse_sigma)):
			for idx in range(k):
				if idx < k-1:
					if idx > 0:
						train = np.concatenate((partitioned_data[0:idx], partitioned_data[idx+1:k]), axis=0)
					else:
						train = partitioned_data[idx+1:k]
				else:
					train = partitioned_data[0:idx]
				validation = partitioned_data[idx:idx+1]
				validation_target = [row[5] for row in partitioned_target[idx:idx+1]]
				gp = GP.GP(train, inverse_beta[i], inverse_sigma[j], 1, 0)
				prediction = gp.predict_feature('Close', test_data=validation)[0]
				errors[i][j] = ac.RMSE(prediction, validation_target)
	return errors

def evaluate_by_feature(results, targets, features, feature):
	num_data = len(results)
	num_days = len(results[0])
	feature = features.index(feature)
	rmse = {}
	for day in range(1, num_days):
		prediction = [data_pt[day][0][feature] for data_pt in results]
		target = [row[feature] for row in targets[1:num_data+1]]
		rmse[str(day)] = ac.RMSE(prediction, target)

	# graph info using MSE with bar graph


def evaluate_by_feature_with_distribution(results, targets, features, feature):
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
		P.figure()
		prediction = [data_pt[day][0][feature] for data_pt in results]
		print "prediction for day:", day, prediction
		target = [row[feature] for row in targets[1:num_data+1]]
		print "target for day:", day, target
		error[str(day)] = ac.percentage_diff(prediction, target)
		error[str(day)] = [diff*100 for diff in error[str(day)]]
		print "diff for day:", day, error[str(day)]
		bins = range(100)
		n, bins, patches = P.hist(error[str(day)], bins, normed=1, histtype='stepfilled')
		P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
	P.show()
	return error