import numpy as np
import Globals
import random
import SVD
import Initialization
import time

def gradientDescent(train,test):
	suffix ='_fixed.npy'
	if not Globals.fixed:
		suffix = '.npy'
	mu = 0
	sigma = 1
	lrate = Globals.lrate
	lamb = 0.02
	if Globals.warmStart:
		print('warm start','./log/NSVD2_v'+Globals.modelIdx+suffix)
		v = np.load('./log/NSVD2_v'+Globals.modelIdx+suffix)
		c = np.load('./log/NSVD2_c'+Globals.modelIdx+suffix)
		d = np.load('./log/NSVD2_d'+Globals.modelIdx+suffix)
	else:
		v = np.empty((Globals.nItems,Globals.k))
		c = np.empty(Globals.nUsers)
		d = np.empty(Globals.nItems)
		for i in range(Globals.nUsers):
			c[i] = random.normalvariate(mu,sigma)
		for i in range(Globals.nItems):
			for j in range(Globals.k):
				v[i,k] = random.normalvariate(mu,sigma)
			d[i] = random.normalvariate(mu,sigma)
	known = train!=0
	globalMean = np.mean(train[mask])

	print('start training')
	prev = 1e9
	curr = 1e8
	t = 0
	startTime = time.time()
	while prev-curr > 1e-9:
		C = np.reshape(c,(Globals.nUsers,1))
		D = np.reshape(d,(1,Globals.nItems))
		C = np.repeat(C,Globals.nItems,axis=1)
		D = np.repeat(D,Globals.nUsers,axis=0)
		A = C+D
		for i in range(Globals.nUsers):
			for j in range(Globals.nItems):
				A[i,j] += np.dot(v[j],np.sum(v[known[i]],axis=0))
		v *= 1-lamb
		c -= lamb*(c+d-globalMean)
		d -= lamb*(c+d-globalMean)
		
		for i in range(Globals.nUsers):
			r = train[i] - A[i]
			term = np.sum(r[known[i]])*lrate*v[j]
			v[j] += np.sum(r[known[i]])*lrate*np.sum(v[known[i]],axis=0)
			c[i] += lrate*np.sum(r[known[i]])
			d[j] += lrate*np.sum(r[known[i]])
			for j in range(Globals.nItems):
				if known[i,j]:
					v[j] += term
		# if t%1000 == 0:
		prev = curr
		curr = SVD.evaluation2(A,test)
		print('t =',t,'score =',curr)
		if t%1000 == 0:
			np.save('./log/NSVD2_v'+Globals.modelIdx+suffix,v)
			np.save('./log/NSVD2_c'+Globals.modelIdx+suffix,c)
			np.save('./log/NSVD2_d'+Globals.modelIdx+suffix,d)
			print('auto save')
		t += 1
	endTime = time.time()
	print('finish training ',int(endTime-startTime),'s')
	np.save('./log/NSVD2_v'+Globals.modelIdx+suffix,v)
	np.save('./log/NSVD2_c'+Globals.modelIdx+suffix,c)
	np.save('./log/NSVD2_d'+Globals.modelIdx+suffix,d)
	# over 5
	mask = A>5
	A[mask] = 5
	# below 1
	mask = A<1
	A[mask] = 1
	score = SVD.evaluation2(A,test)
	print('after clipping score =',score)
	return A

if __name__ == "__main__":
	Initialization.initialization()
	if Globals.fixed:
		train, test = Initialization.readInData2()
		A = gradientDescent(train,test)
		np.save('./log/NSVD2_A_fixed.npy',A)
