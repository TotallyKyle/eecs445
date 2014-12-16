import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import AccuracyEvaluation as ac
import pylab as P
import GP as GP
# put this back in GP

def kfolds(raw_train_data, inverse_beta, inverse_sigma, k=5):
	train_data = raw_train_data[:-1]
	train_target = _map_to_array(raw_train_data[1:])
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
						train = [item for sublist in train for item in sublist]
					else:
						train = partitioned_data[idx+1:k]
						train = [item for sublist in train for item in sublist]
				else:
					train = partitioned_data[0:idx]
					train = [item for sublist in train for item in sublist]
				validation = _map_to_array(partitioned_data[idx:idx+1][0])
				validation_target = [row[5] for row in partitioned_target[idx:idx+1][0]]
				print "compare:", validation[1][5], "and", validation_target[0]
				gp = GP.GP(train, inverse_beta[i], inverse_sigma[j], 1, 0)
				prediction = gp.predict_feature('Close', test_data=np.array(validation))[0]
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
		target = [row[feature] for row in targets[day:num_data+1]]
		print "target for day:", day, target
		error[str(day)] = ac.percentage_diff(prediction, target)
		error[str(day)] = [diff*100 for diff in error[str(day)]]
		print "diff for day:", day, error[str(day)]
		bins = range(100)
		n, bins, patches = P.hist(error[str(day)], bins, normed=1, histtype='stepfilled')
		P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
	P.show()
	return error

def line_graph(results, targets, features, feature):
	num_data = len(results[0])
	num_days = len(results[0][0])
	print "num days", num_days
	feature = features.index(feature)
	mu = results[0]
	s = results[1]
	for day in range(1, num_days):
		x_range = range(0,100)
		plt.figure()
		prediction = np.array([data_pt[day][0][feature] for data_pt in mu])
		#for data_pt in s:
			#print "sigma for idx:", data_pt[day][0][0]
		sigma = np.array([data_pt[day-1][0][0] for data_pt in s])
		target = np.array([row[feature] for row in targets[day:num_data+day]])
		print "prediction:", prediction
		print "target:", target, "len:", len(target)
		#plt.clf()
		plt.plot(x_range, prediction, 'r-')
		plt.plot(x_range, target, 'b-')
		plt.gca().fill_between(x_range, prediction-30*sigma, prediction+30*sigma, color="#dddddd")
		#pl.plot(Xtest, mu, 'r--', lw=2)
		#pl.savefig('predictive.png', bbox_inches='tight')
		plt.title('Mean predictions plus 3 st.deviations')
		#pl.axis([-5, 5, -3, 3])
	plt.show()

def line_graph2(results, target):
	num_data = len(results[0])
	mu = results[0]
	s = results[1]
	x_range = range(0,num_data)
	plt.figure()
	#plt.clf()
	plt.plot(x_range, mu, 'r-')
	plt.plot(x_range, target, 'b-')
	plt.gca().fill_between(x_range, mu-30*s, mu+30*s, color="#dddddd")
	#pl.plot(Xtest, mu, 'r--', lw=2)
	#pl.savefig('predictive.png', bbox_inches='tight')
	plt.title('Mean predictions plus 3 st.deviations')
	#pl.axis([-5, 5, -3, 3])
	plt.show()

def _map_to_array(raw_data):
	mapped_data = []
	print "map to array"
	features = sorted(raw_data[0].keys())
	for idx, row in enumerate(raw_data):
		mapped_data.append([row[key] for key in features])
	return mapped_data   