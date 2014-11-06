import csv
import math
import datetime
import numpy as np

from datetime import *
from datetime import timedelta
from dateutil.parser import parse

def init_parse_alt(filename, feature_names):
	list = []
	feature_col = {}
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
	        for feature, col_num in feature_col.iteritems():
	        	if feature == 'date':
	        		EOD_vals[feature] = parse(row[col_num])
	        	else:
	        		EOD_vals[feature] = row[col_num]
	        list.append(EOD_vals)
	    return list

def init_parse(filename, feature_names, datetime_format='%d.%m.%Y %H:%M:%S.%f'):
	list = []
	feature_col = {}
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
	        list.append(EOD_vals)
	    return list

#join dataset2 onto dataset1 on feature,
#always assume dataset2 is more complete
def join_on_feature(dataset1, dataset2, feature_name='date'):
	data1ptr = 0
	for idx in range(len(dataset2)):
		if dataset1[data1ptr][feature_name].date() == dataset2[idx][feature_name].date():
			dataset1[data1ptr].update(dataset2[idx])
			data1ptr = data1ptr+1
		#else if dataset1[data1ptr][feature_name].date() > dataset2[idx][feature_name].date():
			#do nothing, no matching data in dataset1
	return dataset1

#calculate moving average of past #range# days
#for the specified feature
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
			raw_data.append(data[feature])
		list.append(raw_data)
	return list

#takes a list/list of lists as input
def convert_to_array(data)
	return np.array(data)
