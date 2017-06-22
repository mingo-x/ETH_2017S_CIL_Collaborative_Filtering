# For the given movie j rated by user i, first five predictors
# are empirical probabilities of each rating 1 − 5 for user i.
# The sixth predictor is the mean rating of movie j, after
# subtracting the mean rating of each member.

import Initialization

def p1(inData, n ):
	data = inData.copy()
	for i in range(n):
		c = np.count_nonzero(data[i,:])
		if c==0:
			data[i,:] = 0
			continue
		mask = data[i,:]==1
		t = np.count_nonzero(data[mask])
		p = 1.0*t/c
		data[i,:] = p
	return data


if __name__ == "__main__":
	Initialization.initialization()
	data = Initialization.readInData('./data/data_train.csv')
	pred1 = p1(data,10000)
	print(pred1[0,:])
	print(pred1[1,:])