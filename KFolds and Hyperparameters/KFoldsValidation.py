from	ErrorAnalyser				import  ErrorAnalyser
from 	ModelDataBuilder			import 	ModelDataBuilder
from	sklearn.cross_validation	import	KFold
import	numpy						as		np
import	neurolab 					as		nl

# Evaluates error given labels
# Params:	labeled_predicted
#			labeled_actual
#			list_of_targets - specify which outputs, Ex: Close, Close Tomorrow
# Returns average error using ErrorAnalyser over the K-folds	
def error_helper(labeled_predicted, labeled_actual, list_of_targets):
	analyser = ErrorAnalyser(labeled_predicted, labeled_labeled_actual)
	error = 0;
	for feature_name in list_of_targets:
		error += analyser.AverageErrorForOutput(feature_name)
	return error

def k_folds_validation(raw_inputs, raw_targets, feature_ranges, data_helper, neuron_layer_size_list, num_of_folds, training_func):
	inputs = np.array(raw_inputs)
	targets, target_ranges = np.array(ModelDataBuilder.Normalize(raw_targets))

	kf = KFold(len(inputs), n_folds = num_of_folds)

	average_error = 0;
	counter = 1;
	for train, test in kf:
		inputs_train, inputs_test, targets_train, targets_test = [inputs[i] for i in train], [inputs[j] for j in test], [targets[k] for k in train], [targets[l] for l in test]
		net = nl.net.newff(feature_ranges, neuron_layer_size_list)
		nl.init.midpoint(net.layers[1])

		net.trainf = training_func

		net.train(inputs_train, targets_train, adapt = True, goal = .01, lr = .1)

		predicted = data_helper.Denormalize(net.sim(inputs_test), target_ranges)
		actual    = data_helper.Denormalize(targets_test, target_ranges)

		labeled_predicted = data_helper.MapBackToOutputs(predicted)
		labeled_actual    = data_helper.MapBackToOutputs(actual)

		average_error = (average_error * (counter - 1) + error_helper(labeled_predicted, labeled_actual, ["Close"])) / counter;
		counter += 1;
	return average_error