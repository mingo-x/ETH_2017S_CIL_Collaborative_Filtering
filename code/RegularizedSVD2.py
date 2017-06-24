#add biases to the regularized SVD model, one parameter
#ci for each user and one dj for each movie

import Initialization
import SVD
import numpy as np
import Globals
import random
import time

def biasedRSVD(train,test,k=96):
	# initialization
	# normal distr? N(0,1)
	print('start initialization k =',k)
	lrate = Globals.lrate
	lamb = 0.02
	lamb2 = 0.05
	mu = 0
	sigma = 1
	suffix = '_fixed'+Globals.dataIdx+'.npy'
	if not Globals.fixed:
		suffix = '.npy'
	if Globals.warmStart:
		print('warm start')
		U = np.load('./log/RSVD2_U_'+str(k)+Globals.modelIdx+suffix)
		Vt = np.load('./log/RSVD2_Vt_'+str(k)+Globals.modelIdx+suffix)
		c = np.load('./log/RSVD2_c_'+str(k)+Globals.modelIdx+suffix)
		d = np.load('./log/RSVD2_d_'+str(k)+Globals.modelIdx+suffix)
	else:
		U = np.empty((Globals.nUsers,k))
		Vt = np.empty((k,Globals.nItems))
		c = np.empty(Globals.nUsers)
		d = np.empty(Globals.nItems)
		for i in range(k):
			for j in range(Globals.nUsers):
				U[j,i] = random.normalvariate(mu,sigma)
			for j in range(Globals.nItems):
				Vt[i,j] = random.normalvariate(mu,sigma)
		for i in range(Globals.nUsers):
			c[i] = random.normalvariate(mu,sigma)
		for i in range(Globals.nItems):
			d[i] = random.normalvariate(mu,sigma)
	print('finish initialization')

	print('start SGD')
	startTime = time.time()
	known = train!=0
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

		yp = c[i]+d[j]+U[i,:].dot(Vt[:,j])
		r = train[i,j] - yp
		U[i,:] += lrate*(r*Vt[:,j].T-lamb*U[i,:])
		Vt[:,j] += lrate*(r*U[i,:].T-lamb*Vt[:,j])
		tmp = c[i]+d[j]-globalMean
		c[i] += lrate*(r-lamb2*tmp)
		d[j] += lrate*(r-lamb2*tmp)

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
				break
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

	# end clipping
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

def chooseK(train,test):
	for k in range(30,4,-5):
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
			np.save('./log/RSVD2_A_'+str(Globals.k)+'_fixed.npy',A)
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