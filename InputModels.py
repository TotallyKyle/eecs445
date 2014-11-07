import FeatureParser as parser
import numpy as np


# this generates a sample model whose features are the daily
# close price of the past <num_days>
def timeDelayed(num_days):
	data = parser.init_parse('USDJPY_complete.csv', ['Close'])[0]
	formatted_data = parser.convert_input(data)
	row_num = 0
	container = []
	target = []
	for row in data:
		# ex: if each data point requires data from the past 5 days
		#     we only want to include data starting at day 6
		if row_num > num_days:
			list = parser.time_delayed_data(data, row_num, num_days)
			container.append(list)
			target_val = data[row_num]['Close']
			target.append(target_val)
		row_num = row_num + 1
	minVal = min(target)
	maxVal = max(target)
	target = normalize_target(target, minVal, maxVal)
	return container, target, minVal, maxVal

def timeDelayedFeature(num_days):
	data = parser.init_parse('USDJPY_complete.csv', ['Close'])[0]
	formatted_data = parser.convert_input(data)
	minVal = 9999999
	maxVal = 0
	row_num = 0
	container = []
	target = []
	for row in data:
		# ex: if each data point requires data from the past 5 days
		#     we only want to include data starting at day 6
		if row_num > num_days:
			list = parser.time_delayed_data(data, row_num, num_days)
			dataPt = {}
			dataPt['date'] = data[row_num]['date']
			for i in range(len(list)):
				dataPt[str(i)] = list[i]
				if(list[i] > maxVal):
					maxVal = list[i]
				if(list[i] < minVal):
					minVal = list[i]
			container.append(dataPt)
		row_num = row_num + 1
	return container, [minVal,maxVal]

# the target value needs to be normalized because the
# the NN's output range is restricted to [-1, 1]
def normalize_target(target, minVal, maxVal):
	list = []
	for tar in target:
		normalized_tar = (tar - minVal) / (maxVal-minVal)
		list.append(normalized_tar)
	return list

# denormalize the data after we get the prediction results
# to retrive the original values
def denormalize_target(target, minVal, maxVal):
	list = []
	for tar in target:
		denormalized_tar = (tar * (maxVal-minVal)) + minVal
		list.append(denormalized_tar)
	return list
# def addFeature(dataset, feature_name):