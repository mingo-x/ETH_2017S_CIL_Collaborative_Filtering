#add biases to the regularized SVD model, one parameter
#ci for each user and one dj for each movie

import Initialization
import SVD
import numpy as np
import Globals
import random
import time

def biasedRSVD(data,train,testMask,k=96):
	# initialization
	# normal distr? N(0,1)
	print('start initialization k =',k)
	lrate = Globals.lrate
	lamb = 0.02
	lamb2 = 0.05
	mu = 0
	sigma = 1
	if Globals.warmStart:
		print('warm start')
		U = np.load('./log/RSVD2_U_'+str(k)+Globals.modelIdx+'.npy')
		Vt = np.load('./log/RSVD2_Vt_'+str(k)+Globals.modelIdx+'.npy')
		c = np.load('./log/RSVD2_c_'+str(k)+Globals.modelIdx+'.npy')
		d = np.load('./log/RSVD2_d_'+str(k)+Globals.modelIdx+'.npy')
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
			score = SVD.evaluation(data,A,testMask)
			print('t =',t,'score =',score)
			if score > prev2 and prev2 > prev1:
				break
			prev1 = prev2
			prev2 = score

		# auto save
		if t%500000 == 0:
			np.save('./log/RSVD2_U_'+str(k)+Globals.modelIdx+'.npy',U)
			np.save('./log/RSVD2_Vt_'+str(k)+Globals.modelIdx+'.npy',Vt)
			np.save('./log/RSVD2_c_'+str(k)+Globals.modelIdx+'.npy',c)
			np.save('./log/RSVD2_d_'+str(k)+Globals.modelIdx+'.npy',d)
			print('intermediate result saved')
		t += 1
	endTime = time.time()
	print('finish SGD',int(endTime-startTime),'s')
	np.save('./log/RSVD2_U_'+str(k)+Globals.modelIdx+'.npy',U)
	np.save('./log/RSVD2_Vt_'+str(k)+Globals.modelIdx+'.npy',Vt)
	np.save('./log/RSVD2_c_'+str(k)+Globals.modelIdx+'.npy',c)
	np.save('./log/RSVD2_d_'+str(k)+Globals.modelIdx+'.npy',d)

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
	score = SVD.evaluation(data,A,testMask)
	print('after clipping score =',score)
	return A

def predictionWithCombi(data,k,testMask):
	A1 = np.load('./log/RSVD2_A_'+str(k)+'_clip.npy')
	A2 = np.load('./log/RSVD2_A_'+str(k)+'_2_clip.npy')
	A3 = np.load('./log/RSVD2_A_'+str(k)+'_3_clip.npy')
	A = (A1+A2+A3)/3.0
	score = SVD.evaluation(data,A,testMask)
	print('after combination score =',score)
	return A

if __name__ == "__main__":
	Initialization.initialization()
	data = Initialization.readInData('./data/data_train.csv')
	train, testMask = SVD.splitData(data,10)
	if Globals.predict=='c':
		A = predictionWithCombi(data,Globals.k,testMask)
		np.save('./log/RSVD2_A_'+str(Globals.k)+'_combi.npy',A)
	else:
		A = biasedRSVD(data,train,testMask,Globals.k)
		np.save('./log/RSVD2_A_'+str(Globals.k)+Globals.modelIdx+'_clip.npy',A)
	SVD.writeOutData(A)