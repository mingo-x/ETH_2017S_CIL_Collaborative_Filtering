# using gradient descent with
# regularization and early stopping
# choose K?

import Initialization
import SVD
import numpy as np
import Globals
import random
import time

def SGD(data,train,testMask,k=96):
	# initialization
	# normal distr? N(3,1)
	print('start initialization k =',k)
	lrate = Globals.lrate
	lamb = 0.02
	mu = 0
	sigma = 1
	if Globals.warmStart:
		print('warm start')
		U = np.load('./log/RSVD_U_'+str(k)+'.npy')
		Vt = np.load('./log/RSVD_Vt_'+str(k)+'.npy')
	else:
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
	t = 0
	prev1 = 1000000
	prev2 = 1000000
	while True:
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
		if t%10000 == 0:
			A = U.dot(Vt)
			score = SVD.evaluation(data,A,testMask)
			print('t =',t,'score =',score)
			if score > prev2 and prev2 > prev1:
				break
			prev1 = prev2
			prev2 = score

		# auto save
		if t%500000 == 0:
			np.save('./log/RSVD_U_'+str(k)+'.npy',U)
			np.save('./log/RSVD_Vt_'+str(k)+'.npy',Vt)
			print('intermediate result saved')
		t += 1
	endTime = time.time()
	print('finish SGD',int(endTime-startTime),'s')
	np.save('./log/RSVD_U_'+str(k)+'.npy',U)
	np.save('./log/RSVD_Vt_'+str(k)+'.npy',Vt)

	# clipping
	print('start clipping')
	A = np.zeros((Globals.nUsers,Globals.nItems))
	for m in range(k):
		T = U[:,m:m+1].dot(Vt[m:m+1,:])
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
	Initialization.initialization()
	data = Initialization.readInData('./data/data_train.csv')
	train, testMask = SVD.splitData(data,10)
	A = SGD(data,train,testMask,Globals.k)
	np.save('./log/RSVD_A_'+str(Globals.k)+'.npy',A)
	SVD.writeOutData(A)