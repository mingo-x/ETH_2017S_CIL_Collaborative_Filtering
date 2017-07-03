# DESCRIPTION: This file implements a linear weighted model. Each item j has a weight wj. The rating of item j by user i is linear to the sum of the weights of the user's rated items. Stochastic gradient descent is used to minimize the mean square error and to learn the item weights.

# USAGE: To train the model, run "python3 code/LinearModel.py" and "python3 code/LinearModel.py -d=1". "-d" chooses the datasplit.

import numpy as np
import Globals
import random
import SVD
import Initialization
import time

# SGD
def gradientDescent(train,test,lamb=0.02):
	suffix ='_fixed'+Globals.dataIdx+'.npy'
	if not Globals.fixed:
		suffix = '.npy'
	mu = 0
	sigma = 1
	lrate = Globals.lrate
	# read in previously trained result
	if Globals.warmStart:
		print('warm start','./log/LM_w'+Globals.modelIdx+suffix)
		w = np.load('./log/LM_w'+Globals.modelIdx+suffix)
	# otherwise, initialize randomly
	else:
		w = np.empty(Globals.nItems)
		for i in range(Globals.nItems):
			w[i] = random.normalvariate(mu,sigma)
	m = np.empty(Globals.nItems)
	known = train!=0
	for i in range(Globals.nItems):
		m[i] = np.mean(train[known[:,i],i])
	e = np.empty(Globals.nUsers)
	for i in range(Globals.nUsers):
		e[i] = 1.0/np.sqrt(1+np.count_nonzero(train[i]))

	print('start training')
	prev = 1e9
	curr = 1e8
	t = 0
	A = np.empty((Globals.nUsers,Globals.nItems))
	startTime = time.time()
	# terminate if the validation error stops decreasing
	while prev-curr > 1e-9:
		w *= 1-lamb
		for i in range(Globals.nUsers):
			yp = m.copy()
			yp += e[i]*np.sum(w[known[i]])
			A[i] = yp
			r = train[i] - yp
			term = np.sum(r[known[i]])*lrate*e[i]
			for j in range(Globals.nItems):
				if known[i,j]:
					w[j] += term
		prev = curr
		curr = SVD.evaluation2(A,test)
		print('t =',t,'score =',curr)
		if t%1000 == 0:
			np.save('./log/LM_w'+Globals.modelIdx+suffix,w)
			print('auto save')
		t += 1
	endTime = time.time()
	print('finish training ',int(endTime-startTime),'s')
	np.save('./log/LM_w'+Globals.modelIdx+suffix,w)
	# clipping
	# over 5
	mask = A>5
	A[mask] = 5
	# below 1
	mask = A<1
	A[mask] = 1
	score = SVD.evaluation2(A,test)
	print('after clipping score =',score)
	return A

# choose the regularization parameter
def chooseLamb(train,test):
	for lamb in np.arange(0,0.11,0.01):
		print('lambda =', lamb)
		A = gradientDescent(train,test,lamb)

if __name__ == "__main__":
	Initialization.initialization()
	train, test = Initialization.readInData2(idx=Globals.dataIdx)
	# choose lambda
	if Globals.predict == 'l':
		chooseLamb(train,test)
	# evaluate
	elif Globals.predict == 'e':
		A = np.load('./log/LM_A_fixed'+Globals.dataIdx+'.npy')
		score = SVD.evaluation2(A,test)
		print('score =', score)
	# train and predict
	else:
		A = gradientDescent(train,test,lamb=0.01)
		np.save('./log/LM_A_fixed'+Globals.dataIdx+'.npy',A)