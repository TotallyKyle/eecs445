import numpy.* as np
import FeatureParser as parser

class GP:

	#dataset is an array of dicts
	def __init__(self, data_set):
		self.data_set = data_set
		self.covariance = create_covariance(parser.convert_input(data_set))

	def create_covariance(data_set):
		return kernel(np.array(data_set),np.array(data_set))

	# need to drop data of very last day because we don't have target for that
	def kernel(a, b, inverse_sigma=0.0001, inverse_beta):
	    """ GP squared exponential kernel """
	    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
	    return np.exp(-0.5*inverse_sigma*inverse_sigma)

	def predict(target_name):
		target = create_target(target_name)
		# calculate Cn+1
		# calcaute c
	
	def create_target(self, feature_name):                                               
    	target_data = []
    	for idx, row in enumerate(self.data_set):                                                         
    		if idx + 1 == len(data_set)
    			return np.arraytarget_data
        	target_data.append(dataset[idx + 1][feature_name])
		
	def expand_covariance(Cn_p1, c):