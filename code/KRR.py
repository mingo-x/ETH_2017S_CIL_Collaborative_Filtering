# DESCRIPTION: This file implements an SVD-based kernel ridge regression model. For each user, the V matrix from SVD is fed to the ridge regression as features and the observed ratings as targets. The V matrix is normalized for each item. exp(2(xi.T*xj+1)) is used as the kernel.

# USAGE: To tarin the model, run "python3 code/KRR.py -k=32", "python3 code/KRR.py -k=32 -d=1", "python3 code/KRR.py -i=2 -k=32" and "python3 code/KRR.py -i=2 -k=32 -d=1". "-i" specifies the SVD model used as the base, "-k" specifies the number of dimension used in the SVD model and "-d" chooses the data split.

import Globals
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import Initialization
import SVD

def kernel(x1,x2):
	return np.exp(2*(np.dot(x1,x2)-1))

# count the number of observed ratings for each movie
def topRatedMovies(data):
	for i in range(Globals.nItems):
		count = np.count_nonzero(data[:,i])
		print(count,)

# train and predict
def KRR(data,test, a=0.7):
	suffix = '_fixed'+Globals.dataIdx+'.npy'
	if not Globals.fixed:
		suffix = '.npy'
	known = data!=0
	# read from previous trained result
	if Globals.step == 0:
		A = np.empty((Globals.nUsers,Globals.nItems))
	else:
		A = np.load('./log/KRR'+Globals.modelIdx+'_A_'+str(Globals.k)+suffix)
	Vt = np.load('./log/RSVDF'+Globals.modelIdx+'_V_'+str(Globals.k)+suffix)
	V = Vt.T
	# normalize
	for i in range(Globals.nItems):
		V[i] /= np.linalg.norm(V[i])
	# regression starts here
	for i in range(Globals.step,Globals.nUsers):
		known = data[i]!=0
		y = data[i,known]
		X = V[known]

		clf = KernelRidge(alpha=a,kernel=kernel)
		clf.fit(X, y)
		pred = clf.predict(V)
		A[i] = pred
		if i%10 == 0:
			print('user ',i+1)
			score = SVD.evaluation2(A,test)
			print('score =',score)
		if i%1000 == 0:
			np.save('./log/KRR'+Globals.modelIdx+'_A_'+str(Globals.k)+suffix,A)

	score = SVD.evaluation2(A,test)
	print('alpha =', a, 'test error =',score)
	#clipping
	mask = A>5
	A[mask] = 5
	mask = A<1
	A[mask] = 1
	score = SVD.evaluation2(A,test)
	print('after clipping test error =',score)
	return A

def chooseAlpha(data,test):
	for a in np.arange(0.5,0.9,0.1):
		KRR(data,test,a)

if __name__ == '__main__':
	Initialization.initialization()
	data, test = Initialization.readInData2(idx=Globals.dataIdx)
	# choose the best alpha
	if Globals.predict == 'a':
		chooseAlpha(data,test)
	# train and predict
	else:
		A = KRR(data,test)
		np.save('./log/KRR'+Globals.modelIdx+'_A_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',A)
