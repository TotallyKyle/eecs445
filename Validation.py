#Validation Module
import neurolab as nl
import numpy as np
import InputModels as model
import FeatureParser as parser

def train_nn_set(train_input, train_target, feature_value_range):
	intermediate_layers = []
	# number of hidden layers
	for i in range(4):
		#number of neurons in each hidden layer
		for j in range(2,7):
			intermediate_layers = [j]*i
			for k in range(3):
				if k == 0:
					transfer_func = (i+2)*[nl.trans.TanSig()]
				elif k == 1:
					transfer_func = (i+2)*[nl.trans.LogSig()]
				elif k == 2:
					transfer_func = (i+2)*[nl.trans.SoftMax()]
				layers = [len(feature_value_range)]
				for num in intermediate_layers:
					layers.append(num)
				layers.append(1)
				print "layers: ", layers, " transfer functions: ", transfer_func
				nl.net.newff(feature_value_range,layers, transfer_func)
				error = net.train(train_input, train_target, epochs=500, show=100, goal=0.02)

def MSE(prediction, target):