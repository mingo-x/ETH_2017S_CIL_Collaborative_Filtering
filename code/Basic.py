# DESCRIPTION: This file implements 6 basic predictors. For the given item j rated by user i, the first five predictors are the empirical probabilities of each rating 1 − 5 for user i. The 6th predictor is the mean rating of item j, after subtracting the mean rating of each user.

# USAGE: To train the predictors, run "python3 code/Basic.py" and "python3 code/Basic.py -d=1". "-d" is the option for different training/validation data splits.

import Initialization
import Globals
import numpy as np
import SVD

# basic 1
def p1(inData, n = Globals.nUsers):
	print('basic predictor 1')
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

# basic 2
def p2(inData, n = Globals.nUsers):
	print('basic predictor 2')
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

# basic 3
def p3(inData, n = Globals.nUsers):
	print('basic predictor 3')
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

# basic 4
def p4(inData, n = Globals.nUsers):
	print('basic predictor 4')
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

# basic 5
def p5(inData, n = Globals.nUsers):
	print('basic predictor 5')
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

# basic 6
def p6(inData, nu = Globals.nUsers, ni = Globals.nItems):
	print('basic predictor 6')
	data = inData.copy()
	# mean of users
	meanu = np.empty(nu)
	for i in range(nu):
		mask = data[i,:]!=0
		meanu[i] = np.mean(data[i,mask])
	meani = np.empty(ni)
	for i in range(ni):
		mask = data[:,i]!=0
		# subtract mean of each user
		data[:,i] -= meanu
		# mean of item
		meani[i] = np.mean(data[mask,i])
		data[:,i] = meanu[i]
	return data

if __name__ == "__main__":
	Initialization.initialization()
	if Globals.fixed:
		data, test = Initialization.readInData2(idx=Globals.dataIdx)
	A = p1(data)
	np.save('./log/Basic1_A_fixed'+Globals.dataIdx+'.npy',A)
	score = SVD.evaluation2(A,test)
	print('test error =', score)
	A = p2(data)
	np.save('./log/Basic2_A_fixed'+Globals.dataIdx+'.npy',A)
	score = SVD.evaluation2(A,test)
	print('test error =', score)
	A = p3(data)
	np.save('./log/Basic3_A_fixed'+Globals.dataIdx+'.npy',A)
	score = SVD.evaluation2(A,test)
	print('test error =', score)
	A = p4(data)
	np.save('./log/Basic4_A_fixed'+Globals.dataIdx+'.npy',A)
	score = SVD.evaluation2(A,test)
	print('test error =', score)
	A = p5(data)
	np.save('./log/Basic5_A_fixed'+Globals.dataIdx+'.npy',A)
	score = SVD.evaluation2(A,test)
	print('test error =', score)
	A = p6(data)
	np.save('./log/Basic6_A_fixed'+Globals.dataIdx+'.npy',A)
	score = SVD.evaluation2(A,test)
	print('test error =', score)
