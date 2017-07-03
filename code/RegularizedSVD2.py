# DESCRIPTION: This file implemented a biased l2-norm regularized SVD model with SGD. A bias ci is added for each user i and dj for each item j. The prediction function is U[i]*V[j].T+c[i]+d[j] for user i and item j. SGD terminates when the validation error stops decreasing.

# USAGE: To train the model, run "python3 code/Regularized2.py -k=5" and "python3 code/Regularized2.py -k=5 -d=1". "-k" specifies the number of dimensions for dimension reduction in SVD. "-d" chooses the training/validation data split.

import Initialization
import SVD
import numpy as np
import Globals
import random
import time
from sklearn import linear_model

def biasedRSVD(train,test,k=96):
	# initialization
	print('start initialization k =',k)
	lrate = Globals.lrate
	lamb = 0.02
	lamb2 = 0.05
	mu = 0
	sigma = 1
	suffix = '_fixed'+Globals.dataIdx+'.npy'
	if not Globals.fixed:
		suffix = '.npy'
	# read in previously trained result
	if Globals.warmStart:
		print('warm start')
		U = np.load('./log/RSVD2_U_'+str(k)+Globals.modelIdx+suffix)
		Vt = np.load('./log/RSVD2_Vt_'+str(k)+Globals.modelIdx+suffix)
		c = np.load('./log/RSVD2_c_'+str(k)+Globals.modelIdx+suffix)
		d = np.load('./log/RSVD2_d_'+str(k)+Globals.modelIdx+suffix)
	# otherwise, random initialization
	else:
		U = np.random.rand(Globals.nUsers,k)
		Vt = np.random.rand(k,Globals.nItems)
		c = np.random.rand(Globals.nUsers)
		d = np.random.rand(Globals.nItems)
	known = train!=0
	print('finish initialization')

	print('start SGD')
	startTime = time.time()
	globalMean = np.mean(train[known])
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

		# gradient descent
		yp = c[i]+d[j]+U[i,:].dot(Vt[:,j])
		r = train[i,j] - yp
		Ut = U[i,:].T
		U[i,:] += lrate*(r*Vt[:,j].T-lamb*U[i,:])
		Vt[:,j] += lrate*(r*Ut-lamb*Vt[:,j])
		tmp = lrate*(r - lamb2*(c[i]+d[j]-globalMean))
		c[i] += tmp
		d[j] += tmp

		# evaluation
		if t%10000 == 0:
			A = U.dot(Vt)
			C = np.reshape(c,(Globals.nUsers,1))
			D = np.reshape(d,(1,Globals.nItems))
			C = np.repeat(C,Globals.nItems,axis=1)
			D = np.repeat(D,Globals.nUsers,axis=0)
			A += C+D
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
			np.save('./log/RSVD2_U_'+str(k)+Globals.modelIdx+suffix,U)
			np.save('./log/RSVD2_Vt_'+str(k)+Globals.modelIdx+suffix,Vt)
			np.save('./log/RSVD2_c_'+str(k)+Globals.modelIdx+suffix,c)
			np.save('./log/RSVD2_d_'+str(k)+Globals.modelIdx+suffix,d)
			print('intermediate result saved')
		t += 1
	endTime = time.time()
	print('finish SGD',int(endTime-startTime),'s')
	np.save('./log/RSVD2_U_'+str(k)+Globals.modelIdx+suffix,U)
	np.save('./log/RSVD2_Vt_'+str(k)+Globals.modelIdx+suffix,Vt)
	np.save('./log/RSVD2_c_'+str(k)+Globals.modelIdx+suffix,c)
	np.save('./log/RSVD2_d_'+str(k)+Globals.modelIdx+suffix,d)

	# clipping
	print('start clipping')
	A = U.dot(Vt)
	C = np.reshape(c,(Globals.nUsers,1))
	D = np.reshape(d,(1,Globals.nItems))
	C = np.repeat(C,Globals.nItems,axis=1)
	D = np.repeat(D,Globals.nUsers,axis=0)
	A += C+D
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

# choose the best number of dimensions kept
def chooseK(train,test):
	for k in range(25,4,-5):
		biasedRSVD(train,test,k)

if __name__ == "__main__":
	Initialization.initialization()
	train, test = Initialization.readInData2(idx=Globals.dataIdx)
	# choose the best k
	if Globals.predict == 'k':
		chooseK(train,test)
	# train & predict
	else:
		A = biasedRSVD(train,test,Globals.k)
		np.save('./log/RSVD2_A_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',A)