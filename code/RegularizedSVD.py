# using gradient descent with
# regularization and early stopping
# subtraction of baseline?

import Initialization
import SVD
import numpy as np
import Globals
import random
import time
from sklearn import linear_model

def baseline(train, known):
	nObs = np.count_nonzero(train)
	target = np.reshape(train[known],(nObs,1))

	Basic1_A = np.load('./log/Basic1_A_fixed'+Globals.dataIdx+'.npy')
	Basic2_A = np.load('./log/Basic2_A_fixed'+Globals.dataIdx+'.npy')
	Basic3_A = np.load('./log/Basic3_A_fixed'+Globals.dataIdx+'.npy')
	Basic4_A = np.load('./log/Basic4_A_fixed'+Globals.dataIdx+'.npy')
	Basic5_A = np.load('./log/Basic5_A_fixed'+Globals.dataIdx+'.npy')
	Basic6_A = np.load('./log/Basic6_A_fixed'+Globals.dataIdx+'.npy')

	train = np.append([Basic1_A[known]],[Basic2_A[known]],axis=0)
	train = np.append(train,[Basic3_A[known]],axis=0)
	train = np.append(train,[Basic4_A[known]],axis=0)
	train = np.append(train,[Basic5_A[known]],axis=0)
	train = np.append(train,[Basic6_A[known]],axis=0)
	train = train.T

	test = np.append([Basic1_A.flatten()],[Basic2_A.flatten()],axis=0)
	test = np.append(test,[Basic3_A.flatten()],axis=0)
	test = np.append(test,[Basic4_A.flatten()],axis=0)
	test = np.append(test,[Basic5_A.flatten()],axis=0)
	test = np.append(test,[Basic6_A.flatten()],axis=0)
	test = test.T

	print('start ridge regression')
	startTime = time.time()
	regr = linear_model.Ridge(alpha=0.5, tol=1e-4)
	regr.fit(train, target)
	endTime = time.time()
	print('finish training',int(endTime-startTime),'s')
	print('Coefficients: \n', regr.coef_)

	print('start predicting')
	startTime = time.time()
	A = regr.predict(test)
	endTime = time.time()
	print('finish predicting',int(endTime-startTime),'s',A.shape)
	A = np.reshape(A,(Globals.nUsers,Globals.nItems))

	return A


def SGD(train,test,k=96):
	# initialization
	# normal distr? N(0,1)
	print('start initialization k =',k)
	lrate = Globals.lrate
	lamb = 0.02
	mu = 0
	sigma = 1
	suffix = '_fixed'+Globals.dataIdx+'.npy'
	if not Globals.fixed:
		suffix = '.npy'
	if Globals.warmStart:
		print('warm start')
		U = np.load('./log/RSVD_U_'+str(k)+Globals.modelIdx+suffix)
		Vt = np.load('./log/RSVD_Vt_'+str(k)+Globals.modelIdx+suffix)
	else:
		U = np.empty((Globals.nUsers,k))
		Vt = np.empty((k,Globals.nItems))
		for i in range(k):
			for j in range(Globals.nUsers):
				U[j,i] = random.normalvariate(mu,sigma)
			for j in range(Globals.nItems):
				Vt[i,j] = random.normalvariate(mu,sigma)
	known = train!=0
	base = baseline(train,known)
	train -= base
	print('finish initialization')

	print('start SGD')
	startTime = time.time()
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
			A = U.dot(Vt)+base
			score = SVD.evaluation2(A,test)
			print('t =',t,'score =',score)
			if score > prev2 and prev2 > prev1:
				break
			prev1 = prev2
			prev2 = score

		# auto save
		if t%500000 == 0:
			np.save('./log/RSVD_U_'+str(k)+Globals.modelIdx+suffix,U)
			np.save('./log/RSVD_Vt_'+str(k)+Globals.modelIdx+suffix,Vt)
			print('intermediate result saved')
		t += 1
	endTime = time.time()
	print('finish SGD',int(endTime-startTime),'s')
	np.save('./log/RSVD_U_'+str(k)+Globals.modelIdx+suffix,U)
	np.save('./log/RSVD_Vt_'+str(k)+Globals.modelIdx+suffix,Vt)

	# clipping
	print('start clipping')
	A = U.dot(Vt)+base
	# over 5
	mask = A>5
	A[mask] = 5
	# below 1
	mask = A<1
	A[mask] = 1
	# A = np.zeros((Globals.nUsers,Globals.nItems))
	# for m in range(k):
	# 	T = U[:,m:m+1].dot(Vt[m:m+1,:])
	# 	A += T
	# 	# over 5
	# 	mask = A>5
	# 	A[mask] = 5
	# 	# below 1
	# 	mask = A<1
	# 	A[mask] = 1
	print('finish clipping')
	score = SVD.evaluation2(A,test)
	print('after clipping score =',score)
	return A

def chooseK(train,test):
	for k in range(15,26,5):
		SGD(train,test,k)


def predictionWithCombi(k,test):
	A1 = np.load('./log/RSVD_A_'+str(k)+'_clip.npy')
	A2 = np.load('./log/RSVD_A_'+str(k)+'_2_clip.npy')
	A3 = np.load('./log/RSVD_A_'+str(k)+'_3_clip.npy')
	A = (A1+A2+A3)/3.0
	score = SVD.evaluation2(A,test)
	print('after combination score =',score)
	return A

if __name__ == "__main__":
	Initialization.initialization()
	if Globals.fixed:
		train, test = Initialization.readInData2(idx=Globals.dataIdx)
		if Globals.predict == 'k':
			chooseK(train,test)
		else:
			A = SGD(train,test,Globals.k)
			np.save('./log/RSVD_A_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',A)
	else:
		data = Initialization.readInData('./data/data_train.csv')
		train, test = SVD.splitData(data,10)
		if Globals.predict=='c':
			A = predictionWithCombi(Globals.k,test)
			np.save('./log/RSVD_A_'+str(Globals.k)+'_combi.npy',A)
		else:
			A = SGD(train,test,Globals.k)
			np.save('./log/RSVD_A_'+str(Globals.k)+Globals.modelIdx+'.npy',A)
		SVD.writeOutData(A)