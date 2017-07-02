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

def sigmoid(x):
	return math.exp(-np.logaddexp(0, -x))

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
	# base = SVD.baseline(train,known)
	# train -= base
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
			A = U.dot(Vt)
			score = SVD.evaluation2(A,test)
			print('t =',t,'score =',score)
			if score > prev2 and prev2 > prev1:
				if lrate <= 1e-5:
					break
				else:
					lrate *= 0.1
					print('learning rate =',lrate)
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
	A = U.dot(Vt)
	# over 5
	mask = A>5
	A[mask] = 5
	# below 1
	mask = A<1
	A[mask] = 1
	
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
		elif Globals.predict == 'e':
			A = np.load('./log/RSVD_A_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy')
			score = SVD.evaluation2(A,test)
			print('score =', score)
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