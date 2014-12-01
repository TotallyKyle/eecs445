#Validation Module
import neurolab as nl
import numpy as np
import InputModels as model
import FeatureParser as parser

# 
def train_nn_set(train_input, train_target, feature_value_range):
	nets = []
	intermediate_layers = []
	# number of hidden layers
	#4
	for i in range(2):
		#number of neurons in each hidden layer
		#(2,7)
		for j in range(2,3):
			intermediate_layers = [j]*i
			for k in range(3):
				layers = []
				if k == 0:
					transfer_func = (i+2)*[nl.trans.TanSig()]
				elif k == 1:
					transfer_func = (i+2)*[nl.trans.LogSig()]
				elif k == 2:
					transfer_func = (i+2)*[nl.trans.SoftMax()]
				layers.append(len(feature_value_range))
				for idx in range(len(intermediate_layers)):
					layers.append(intermediate_layers[idx])
				layers.append(1)
				print "layers: ", layers, " transfer functions: ", transfer_func
				nets.append(nl.net.newff(feature_value_range,layers, transfer_func))
				for i in range(len(intermediate_layers)):
					nl.init.midpoint(nets[len(nets)-1].layers[i])
				error = nets[len(nets)-1].train(train_input, train_target, epochs=500, show=100, goal=0.02)
	return nets

def k_folds(k, train_input, train_target, feature_value_range, target_min, target_max):
	train_size = len(train_input)
	segment_size = train_size / k
	# we currently train 36 NNs with different configs in train_nn_set()
	errors = [0]*36
	#call train_nn_set k times
	for i in range(k):
		partition = get_partition(k, i, train_input, train_target)
		valInp = partition[0]
		valTar = partition[1]
		inp = partition[2]
		tar = partition[3]
		nets = train_nn_set(inp, tar, feature_value_range)
		kth_errors = validate_nn_set(nets, valInp, valTar, target_min, target_max)
		errors = [errors + kth_errors for errors, kth_errors in zip(errors, kth_errors)]
	for error in errors:
		error = error / k
	return errors

def validate_nn_set(nets, validation_input, validation_target, target_min, target_max):
	msef = nl.error.MSE()
	for net in nets:
		target = list(validation_target)
		prediction = model.denormalize_target(net.sim(validation_input).reshape(1,len(validation_input)).tolist()[0],target_min, target_max)
		error = [prediction - target for prediction, target in zip(prediction, target)]
		errors[i] = errors[i] + msef(error)
	return errors

def get_partition(total, k, train_input, train_target):
	train_size = len(train_input)
	segment_size = train_size / total
	valInp = train_input[k*segment_size:(k+1)*segment_size]
	valTar = train_target[k*segment_size:(k+1)*segment_size]
	inp = list(train_input)
	tar = list(train_target)
	inp[k*segment_size:(k+1)*segment_size] = []
	tar[k*segment_size:(k+1)*segment_size] = []
	return valInp, valTar, inp, tar

