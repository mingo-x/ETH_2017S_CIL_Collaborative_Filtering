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
	w = np.empty(Globals.nItems)
	m = np.empty(Globals.nItems)
	known = train!=0
	for i in range(Globals.nItems):
		w[i] = random.normalvariate(mu,sigma)
		m[i] = np.mean(train[known[:,i],i])
	e = np.empty(Globals.nUsers)
	for i in range(Globals.nUsers):
		e[i] = 1.0/np.sqrt(1+np.count_nonzero(data[i]))

	print('start training')
	prev0 = 1e10
	prev1 = 1e9
	curr = 1e8
	t = 0
	A = np.empty((Globals.nUsers,Globals.nItems))
	startTime = time.time()
	while prev1>prev0 and prev0>curr:
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
		if t%1000 == 0:
			prev1 = prev0
			prev0 = curr
			curr = SVD.evaluation(data,A,testMask)
			print('t =',t,'score =',curr)
		if t%10000 == 0:
			np.save('./log/LM_w.npy',w)
			print('auto save')
		t += 1
	endTime = time.time()
	print('finish training ',int(endTime-startTime),'s')
	np.save('./log/LM_w.npy',w)
	# over 5
	mask = A>5
	A[mask] = 5
	# below 1
	mask = A<1
	A[mask] = 1
	score = SVD.evaluation(data,A,testMask)
	print('after clipping score =',score)
	return A

if __name__ == "__main__":
	Initialization.initialization()
	data = Initialization.readInData('./data/data_train.csv')
	train, testMask = SVD.splitData(data,10)
	A = gradientDescent(data,train,testMask)
	np.save('./log/LM_A.npy',A)
	SVD.writeOutData(A)
