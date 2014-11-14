import csv
import math
import datetime
import numpy as np

from datetime import *
from datetime import timedelta
from dateutil.parser import parse

# this parser works specifically with the format
# that is in initial_features_edited.csv
def init_parse_alt(filename, feature_names):
	list = []
	feature_col = {}
	minVal = 9999999
	maxVal = 0
	with open(filename, 'rb') as f:
	    reader = csv.reader(f)
	    line = reader.next()
	    reader.next() #ignore extraneous lines
	    reader.next() #ignore extraneous lines
	    feature_col['date'] = 0
	    for feature in feature_names:
	    	feature_col[feature] = line.index(feature) + 1
	    #return feature_col

	    for row in reader:
	    	EOD_vals = {}
	    	invalid = False
	        for feature, col_num in feature_col.iteritems():
	        	if feature == 'date':
	        		EOD_vals[feature] = parse(row[col_num])
	        	else:
	        		if row[col_num] == '':
	        			invalid = True
	        		else:
	        			EOD_vals[feature] = float(row[col_num])
	        			if(EOD_vals[feature] > maxVal):
	        				maxVal = EOD_vals[feature]
	        			if(EOD_vals[feature] < minVal):
	        				minVal = EOD_vals[feature]
	    	if invalid == False:
	        	list.append(EOD_vals)
	    return list, minVal, maxVal

# this parser works specifically with the format
# that is in USD_JPY_sample.csv
# more exchange rate historical data can be downloaded from
# http://www.forexrate.co.uk/forexhistoricaldata.php
def init_parse(filename, feature_names, datetime_format='%d.%m.%Y %H:%M:%S.%f'):
	list = []
	feature_col = {}
	minVal = 9999999
	maxVal = 0
	with open(filename, 'rb') as f:
	    reader = csv.reader(f)
	    line = reader.next()
	    feature_col['date'] = 0
	    for feature in feature_names:
	    	feature_col[feature] = line.index(feature)
	    #return feature_col

	    for row in reader:
	    	EOD_vals = {}
	        for feature, col_num in feature_col.iteritems():
	        	if feature == 'date':
	        		#EOD_vals[feature] = parse(row[col_num])
	        		EOD_vals[feature] = datetime.strptime(row[col_num], datetime_format)
	        	else:
	        		EOD_vals[feature] = float(row[col_num])
	        		if(EOD_vals[feature] > maxVal):
	        			maxVal = EOD_vals[feature]
	        		if(EOD_vals[feature] < minVal):
	        			minVal = EOD_vals[feature]
	        list.append(EOD_vals)
	    return list, minVal, maxVal

# joins 2 datasets together by joining on the smaller
# of the 2 sets because some fields have missing dates
def join_on_minimum(dataset1, dataset2, feature_name='date'):
	if len(dataset1) > len(dataset2):
		larger = dataset1
		smaller = dataset2
	else:
		larger = dataset2
		smaller = dataset1
	smallerIdx = 0
	while smaller[smallerIdx][feature_name].date() < larger[0][feature_name].date():
		smallerIdx = smallerIdx + 1
	result = []
	for idx in range(len(larger)):
		if smaller[smallerIdx][feature_name].date() == larger[idx][feature_name].date():
			smaller[smallerIdx].update(larger[idx])
			result.append(smaller[smallerIdx])
			smallerIdx = smallerIdx+1
			if smallerIdx == len(smaller):
				break
	return result

def match_target_to_data(target, data, feature_name='date'):
	dataIdx = 0
	matched_target = []
	for idx in range(len(target)):
		if data[dataIdx][feature_name].date() == target[idx][feature_name].date():
			matched_target.append(target[idx])
			dataIdx = dataIdx+1
			if dataIdx == len(data):
				break
	return matched_target

def add_feature_to_data_alt(dataset, feature_source, feature_name):
	new_data = init_parse_alt(feature_source, [feature_name])
	new_feature_data = new_data[0]
	minVal = new_data[1]
	maxVal = new_data[2]

	if len(dataset) > 0:
		dataset = join_on_minimum(dataset, new_feature_data)
	else:
		dataset = new_feature_data
	return dataset, [minVal, maxVal]

def add_feature_to_data(dataset, feature_source, feature_name):
	new_data = init_parse(feature_source, [feature_name])
	new_feature_data = new_data[0]
	minVal = new_data[1]
	maxVal = new_data[2]

	if len(dataset) > 0:
		dataset = join_on_minimum(dataset, new_feature_data)
	else:
		dataset = new_feature_data
	return dataset, [minVal, maxVal]

def moving_average(dataset, target_row_num, range_len, feature='Close'):
	total = 0
	for i in range(range_len):
		total = total + dataset[target_row_num - i]
	return total / range_len

def time_delayed_data(dataset, target_row_num, range_len, feature='Close'):
	list = []
	for i in range(range_len):
		list.append(dataset[target_row_num - i][feature])
	return list
def get_data_by_date_range(dataset, start_date, num_days):
	start = parse (start_date)
	duration = timedelta(days=num_days)
	print "start date is", start
	print "duration is", duration

def segmentation(corpus, train, validation, test):
	size = len(corpus)
	train_size = int(math.floor(size*train))
	test_size = int(math.floor(size*test))
	validation_size = int(math.floor(size*validation))
	trainLower = 0
	trainUpper = train_size
	validationLower = trainUpper+1
	validationUpper = validationLower + validation_size
	testLower = validationUpper+1
	testUpper = size

	train_set = corpus[trainLower:trainUpper]
	validation_set = corpus[validationLower:validationUpper]
	test_set = corpus[testLower:testUpper]
	return train_set, validation_set, test_set

def convert_input(dataset):
	list = [];
	for data in dataset:
		raw_data = []
		for feature in data:
			if feature != 'date':
				raw_data.append(data[feature])
		list.append(raw_data)
	return list

def convert_feature_value_range(feature_value_ranges):
	ranges = []
	for feature in feature_value_ranges:
			if feature != 'date':
				ranges.append(feature_value_ranges[feature])
	return ranges

#takes a list/list of lists as input
def convert_to_array(data):
	return np.array(data)
