# For the given movie j rated by user i, first five predictors
# are empirical probabilities of each rating 1 − 5 for user i.
# The sixth predictor is the mean rating of movie j, after
# subtracting the mean rating of each member.

import Initialization
import Globals
import numpy as np

def p1(inData, n = Globals.nUsers):
	data = inData.copy()
	for i in range(n):
		c = np.count_nonzero(data[i,:])
		if c==0:
			data[i,:] = 0
			continue
		mask = data[i,:]==1
		t = np.count_nonzero(data[i,mask])
		p = 1.0*t/c
		data[i,:] = p
	return data

def p2(inData, n = Globals.nUsers):
	data = inData.copy()
	for i in range(n):
		c = np.count_nonzero(data[i,:])
		if c==0:
			data[i,:] = 0
			continue
		mask = data[i,:]==2
		t = np.count_nonzero(data[i,mask])
		p = 1.0*t/c
		data[i,:] = p
	return data

def p3(inData, n = Globals.nUsers):
	data = inData.copy()
	for i in range(n):
		c = np.count_nonzero(data[i,:])
		if c==0:
			data[i,:] = 0
			continue
		mask = data[i,:]==3
		t = np.count_nonzero(data[i,mask])
		p = 1.0*t/c
		data[i,:] = p
	return data

def p4(inData, n = Globals.nUsers):
	data = inData.copy()
	for i in range(n):
		c = np.count_nonzero(data[i,:])
		if c==0:
			data[i,:] = 0
			continue
		mask = data[i,:]==4
		t = np.count_nonzero(data[i,mask])
		p = 1.0*t/c
		data[i,:] = p
	return data

def p5(inData, n = Globals.nUsers):
	data = inData.copy()
	for i in range(n):
		c = np.count_nonzero(data[i,:])
		if c==0:
			data[i,:] = 0
			continue
		mask = data[i,:]==5
		t = np.count_nonzero(data[i,mask])
		p = 1.0*t/c
		data[i,:] = p
	return data

def p6(inData, nu = Globals.nUsers, ni = Globals.nItems):
	data = inData.copy()
	# mean of users
	meanu = np.empty(nu)
	for i in range(nu):
		mask = data[i,:]!=0
		meanu[i] = np.mean(data[i,mask])
	meani = np.empty(ni)
	for i in range(ni):
		mask = data[:,i]!=0
		meani[i] = np.mean(data[mask,i])
	for i in range(nu):
		for j in range(ni):
			data[i,j] = meani[j] - meanu[i]
	return data

if __name__ == "__main__":
	Initialization.initialization()
	data = Initialization.readInData('./data/data_train.csv')
	# pred1 = p1(data)
	# pred2 = p2(data)
	# pred3 = p3(data)
	# pred4 = p4(data)
	# pred5 = p5(data)
	# print(pred1[0,0], pred2[0,0], pred3[0,0], pred4[0,0], pred5[0,0])
	pred6 = p6(data)
	print(pred6[0,:])
