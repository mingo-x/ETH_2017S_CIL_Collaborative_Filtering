#add biases to the regularized SVD model, one parameter
#ci for each user and one dj for each movie

import Initialization
import SVD
import numpy as np
import Globals
import random
import time
from sklearn import linear_model
import math

def sigmoid(x):
	return math.exp(-np.logaddexp(0, -x))

def biasedRSVD(train,test,k=96):
	# initialization
	# normal distr? N(0,1)
	print('start initialization k =',k)
	lrate = Globals.lrate
	lamb = 0.02
	lamb2 = 0.05
	suffix = '_fixed'+Globals.dataIdx+'.npy'
	step = 10000
	if Globals.step != 0:
		step = Globals.step
	if not Globals.fixed:
		suffix = '.npy'
	if Globals.warmStart:
		print('warm start')
		U = np.load('./log/NRSVD2_U_'+str(k)+Globals.modelIdx+suffix)
		Vt = np.load('./log/NRSVD2_Vt_'+str(k)+Globals.modelIdx+suffix)
		c = np.load('./log/NRSVD2_c_'+str(k)+Globals.modelIdx+suffix)
		d = np.load('./log/NRSVD2_d_'+str(k)+Globals.modelIdx+suffix)
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

		x = sigmoid(c[i]+d[j]+U[i,:].dot(Vt[:,j]))
		yp = 4*x+1
		r = train[i,j] - yp
		eff = 2*r*4*x*(1-x)
		Ut = U[i,:].T
		U[i,:] += lrate*(eff*Vt[:,j].T-lamb*U[i,:])
		Vt[:,j] += lrate*(eff*Ut-lamb*Vt[:,j])
		tmp = (eff-(c[i]+d[j]-globalMean)*lamb2)*lrate
		c[i] += tmp
		d[j] += tmp

		# evaluation
		if t%step == 0:
			A = U.dot(Vt)
			C = np.reshape(c,(Globals.nUsers,1))
			D = np.reshape(d,(1,Globals.nItems))
			C = np.repeat(C,Globals.nItems,axis=1)
			D = np.repeat(D,Globals.nUsers,axis=0)
			A += C+D
			A = 4*np.exp(-np.logaddexp(0,-A))+1
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
			np.save('./log/NRSVD2_U_'+str(k)+Globals.modelIdx+suffix,U)
			np.save('./log/NRSVD2_Vt_'+str(k)+Globals.modelIdx+suffix,Vt)
			np.save('./log/NRSVD2_c_'+str(k)+Globals.modelIdx+suffix,c)
			np.save('./log/NRSVD2_d_'+str(k)+Globals.modelIdx+suffix,d)
			print('intermediate result saved')
		t += 1
	endTime = time.time()
	print('finish SGD',int(endTime-startTime),'s')
	np.save('./log/NRSVD2_U_'+str(k)+Globals.modelIdx+suffix,U)
	np.save('./log/NRSVD2_Vt_'+str(k)+Globals.modelIdx+suffix,Vt)
	np.save('./log/NRSVD2_c_'+str(k)+Globals.modelIdx+suffix,c)
	np.save('./log/NRSVD2_d_'+str(k)+Globals.modelIdx+suffix,d)

	# end clipping
	print('start clipping')
	A = U.dot(Vt)
	C = np.reshape(c,(Globals.nUsers,1))
	D = np.reshape(d,(1,Globals.nItems))
	C = np.repeat(C,Globals.nItems,axis=1)
	D = np.repeat(D,Globals.nUsers,axis=0)
	A += C+D
	A = 4*np.exp(-np.logaddexp(0,-A))+1
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
	for k in range(25,4,-5):
		biasedRSVD(train,test,k)

def predictionWithCombi(k,test):
	A1 = np.load('./log/RSVD2_A_'+str(k)+'_clip.npy')
	A2 = np.load('./log/RSVD2_A_'+str(k)+'_2_clip.npy')
	A3 = np.load('./log/RSVD2_A_'+str(k)+'_3_clip.npy')
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
			A = biasedRSVD(train,test,Globals.k)
			np.save('./log/NRSVD2_A_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',A)
	else:
		data = Initialization.readInData('./data/data_train.csv')
		train, test = SVD.splitData(data,10)
		if Globals.predict=='c':
			A = predictionWithCombi(Globals.k,test)
			np.save('./log/RSVD2_A_'+str(Globals.k)+'_combi.npy',A)
		else:
			A = biasedRSVD(train,test,Globals.k)
			np.save('./log/RSVD2_A_'+str(Globals.k)+Globals.modelIdx+'_clip.npy',A)
		SVD.writeOutData(A)