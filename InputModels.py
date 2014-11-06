import FeatureParser as parser
import numpy as np

def timeDelayed(num_days):
	data = parser.init_parse('USD_JPY_sample.csv', ['Close'])
	formatted_data = parser.convert_input(data)
	row_num = 0
	container = []
	target = []
	for row in data:
		if row_num > num_days:
			list = parser.time_delayed_data(data, row_num, num_days)
			container.append(list)
			target_val = data[row_num]['Close']
			target.append(target_val)
		row_num = row_num + 1
	minVal = min(target)
	maxVal = max(target)
	target = normalize_target(target, minVal, maxVal)
	print min(target)
	return container, target, minVal, maxVal


def normalize_target(target, minVal, maxVal):
	list = []
	for tar in target:
		normalized_tar = (tar - minVal) / (maxVal-minVal)
		list.append(normalized_tar)
	return list

def denormalize_target(target, minVal, maxVal):
	list = []
	for tar in target:
		denormalized_tar = (tar * (maxVal-minVal)) + minVal
		print "tar is: ", tar, " minval is: ", minVal, "maxval is: ", maxVal
		print "denormalized_tar is: ", denormalized_tar
		list.append(denormalized_tar)
	return list
# def addFeature(dataset, feature_name):