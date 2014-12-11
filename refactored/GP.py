import numpy as np
import FeatureParser as parser
from   sklearn import cross_validation

# TODO : remove all instances of 5 to parametrize
# TODO : redo math to avoid re calculation of cholesky and covariance
class GP:

	#dataset is an array of dicts
	def __init__(self, data_set, inverse_beta=0.01, inverse_sigma=0.0001, train=0.7, test=0.3):
		self.raw_data = data_set
		self.train = train
		self.test = test
		self.inverse_sigma = inverse_sigma
		self.inverse_beta = inverse_beta
		self.features = sorted(self.raw_data[30].keys())
		converted = self._map_to_array()
		self.data_set = np.array(converted[0:-1]) # 5 is the number of days we're going back, -2 is a hack for ndarray
		self.train_data, validation, self.test_data = parser.segmentation(self.data_set, train, 0, test)
		self.covariance = self.create_covariance(self.train_data)
		
	
	def create_covariance(self, data_set):
		return self.kernel(data_set, data_set)

	# breakout kernel function into lambda functor
	def kernel(self, a, b, inverse_beta=0.01, inverse_sigma=0.0001):
	    """ GP squared exponential kernel """
	    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
	    return np.exp(-0.5*inverse_sigma*inverse_sigma*sqdist)

	def predict_feature(self, target_name, train_data=None, train_target = None, test_data=None, covariance=None):
		if train_data is None:
			train_data = self.train_data
		if train_target is None:
			train_target = self.create_target(target_name)[0]
		if test_data is None:
			test_data = self.test_data
		if covariance is None:
			covariance = self.covariance
		
		# compute the mean at our test points.
		L =  np.linalg.cholesky(covariance + self.inverse_beta*np.eye(covariance.shape[0]))
		Lk = np.linalg.solve(L, self.kernel(train_data, test_data))
		mu = np.dot(Lk.T, np.linalg.solve(L, train_target))

		# compute the variance at our test points.
		K_ = self.kernel(test_data, test_data)
		s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
		s = np.sqrt(s2)
		return mu, s

	def predict(self, num_days, test_data=None):
		if test_data is None:
			test_data = self.test_data[0:5]
		result_set = []
		result_std_dev = []
		data_left = len(test_data)
		for data_point in test_data:
			print "********************************"
			print "       Data remaining: ", data_left
			print "********************************" 
			data_left -= 1
			prediction_set = []
			prediction_std_set = []
			for i in range(num_days):
				print "    ****************************"
				print "		     Day:", i
				print "    ****************************"
				predicted_values = []
				predicted_stds = []
				for feature in self.features:
					covariance = self.covariance if not prediction_set else self.expand_kernel(prediction_set[:-1])
					train_data = self.train_data if not prediction_set else self.expand_data(prediction_set[:-1])
					train_target = None if not prediction_set else self.create_target(feature, prediction_set[:-1])[0]
					test_point = data_point.reshape(1,len(self.features)) if i == 0 else prediction_set[-1]
					prediction = self.predict_feature(feature, train_data, train_target, test_point, covariance)
					predicted_values.append(prediction[0])
					predicted_stds.append(prediction[1])
				if i == 0:
					prediction_set.append(test_point)
				prediction_set.append(np.array(predicted_values).reshape(1,len(self.features)))
				prediction_std_set.append(np.array(predicted_stds).reshape(1,len(self.features)))
				print prediction_set[-1]
			result_set.append(prediction_set)
			result_std_dev.append(prediction_std_set)
		return result_set, result_std_dev
	def expand_kernel(self, extra_data):
		expanded_data = self.expand_data(extra_data)
		return self.kernel(expanded_data, expanded_data)

	# return expanded train_data, train_target, kernel
	def expand_data(self, extra_data):
		extra_data = np.array(extra_data).reshape(len(extra_data),len(self.features))
		return np.concatenate((self.train_data, extra_data), axis=0)

	def create_target(self, feature_name, extra = None):                                               
		target_data = []
		for idx, row in enumerate(self.raw_data):
			if idx < 30:
				continue                                                         
			if idx + 1 == len(self.raw_data):
				break
			target_data.append(self.raw_data[idx + 1][feature_name])
		train_target, validation, test_target = parser.segmentation(target_data, self.train, 0, self.test)
		if extra is not None:
			idx = self.features.index(feature_name)
			extra = [item for sublist in extra for item in sublist]
			train_target = np.append(train_target,[row[idx] for row in extra]) 
 		return np.array(train_target), np.array(test_target)	

	def _map_to_array(self):
		mapped_data = []
		for idx, row in enumerate(self.raw_data):
			if idx < 30:
				continue
			mapped_data.append([row[key] for key in self.features])
		return mapped_data   