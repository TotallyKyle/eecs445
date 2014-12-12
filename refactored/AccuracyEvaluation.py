def RMSE(prediction, target):
	diff = [a_i - b_i for a_i, b_i in zip(prediction, target)]
	squared_diff = [i**2 for i in diff]
	num_data = len(prediction)
	return (sum(squared_diff) / num_data)**0.5

def percentage_diff(prediction, target):
	diff = [a_i - b_i for a_i, b_i in zip(prediction, target)]
	normalized = [abs(diff_i/target_i) for diff_i, target_i in zip (diff, target)]
	return normalized