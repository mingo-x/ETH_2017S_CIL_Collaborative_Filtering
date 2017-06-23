import numpy as np
import Globals
import random
import SVD
import Initialization
import time

def gradientDescent(data,train,testMask):
	mu = 0
	sigma = 1
	lrate = Globals.lrate
	lamb = 0.02
	if Globals.warmStart:
		w = np.load('./log/LM_w'+Globals.modelIdx+'.npy')
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
		# if t%1000 == 0:
		prev = curr
		curr = SVD.evaluation(data,A,testMask)
		print('t =',t,'score =',curr)
		if t%1000 == 0:
			np.save('./log/LM_w'+Globals.modelIdx+'.npy',w)
			print('auto save')
		t += 1
	endTime = time.time()
	print('finish training ',int(endTime-startTime),'s')
	np.save('./log/LM_w'+Globals.modelIdx+'.npy',w)
	# over 5
	mask = A>5
	A[mask] = 5
	# below 1
	mask = A<1
	A[mask] = 1
	score = SVD.evaluation(data,A,testMask)
	print('after clipping score =',score)
	return A

def predictionWithCombi(data,testMask):
	A1 = np.load('./log/LM_A.npy')
	A2 = np.load('./log/LM_A_2.npy')
	A3 = np.load('./log/LM_A_3.npy')
	A = (A1+A2+A3)/3.0
	score = SVD.evaluation(data,A,testMask)
	print('after combination score =',score)
	return A

if __name__ == "__main__":
	Initialization.initialization()
	data = Initialization.readInData('./data/data_train.csv')
	train, testMask = SVD.splitData(data,10)
	if Globals.predict=='c':
		A = predictionWithCombi(data,testMask)
		np.save('./log/LM_A_combi.npy',A)
	else:
		A = gradientDescent(data,train,testMask)
		np.save('./log/LM_A'+Globals.modelIdx+'.npy',A)
	SVD.writeOutData(A)
