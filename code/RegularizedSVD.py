# using gradient descent with
# regularization and early stopping

import Initialization
import SVD
import numpy as np
import Globals
import random
import time

def SGD(data,train,testMask,k=96):
	# initialization
	# normal distr? N(0,1)
	print('start initialization')
	lrate = 0.001
	lamb = 0.02
	mu = 0
	sigma = 1
	U = np.empty((Globals.nUsers,k))
	Vt = np.empty((k,Globals.nItems))
	for i in range(k):
		for j in range(Globals.nUsers):
			U[j,i] = random.normalvariate(mu,sigma)
		for j in range(Globals.nItems):
			Vt[i,j] = random.normalvariate(mu,sigma)
	print('finish initialization')

	print('start SGD')
	startTime = time.time()
	known = train!=0
	for t in range(1000000):
		# random choice of training sample
		i = random.randint(0,Globals.nUsers-1)
		j = random.randint(0,Globals.nItems-1)
		while not known[i,j]:
			i = random.randint(0,Globals.nUsers-1)
			j = random.randint(0,Globals.nItems-1)

		yp = U[i,:].dot(Vt[:,j])
		r = train[i,j] - yp
		U[i,:] += lrate*(r*Vt[:,j].T-lamb*U[i,:])
		Vt[:,j] += lrate*(r*U[i,:].T-lamb*Vt[:,j])

		# evaluation
		if t%1000 == 0:
			A = U.dot(Vt)
			score = SVD.evaluation(data,A,testMask)
			endTime = time.time()
			print('t =',t,'score =',score, 'time =', int(endTime-startTime), 's')
			startTime = time.time()
	print('finish SGD')

	# clipping
	print('start clipping')
	A = np.zeros((Globals.nUsers,Globals.nItems))
	for m in range(k):
		T = U[:,m].dot(Vt[m,:])
		A += T
		# over 5
		mask = A>5
		A[mask] = 5
		# below 1
		mask = A<1
		A[mask] = 1
	print('finish clipping')
	score = SVD.evaluation(data,A,testMask)
	print('after clipping score =',score)
	return A


if __name__ == "__main__":
	data = Initialization.readInData('./data/data_train.csv')
	train, testMask = SVD.splitData(data,10)
	A = SGD(data,train,testMask)
	SVD.writeOutData(A)