import Globals
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import Initialization
import SVD

def kernel(x1,x2):
	return np.exp(2*(np.dot(x1,x2)-1))

def topRatedMovies(data):
	for i in range(Globals.nItems):
		count = np.count_nonzero(data[:,i])
		print(count,)

def KRR(data,test, a=0.7):
	suffix = '_fixed'+Globals.dataIdx+'.npy'
	if not Globals.fixed:
		suffix = '.npy'
	known = data!=0
	# base = SVD.baseline(data,known)
	# data -= base
	if Globals.step == 0:
		# A = base.copy()
		A = np.empty((Globals.nUsers,Globals.nItems))
	else:
		A = np.load('./log/KRR'+Globals.modelIdx+'_A_'+str(Globals.k)+suffix)
	Vt = np.load('./log/RSVDF'+Globals.modelIdx+'_V_'+str(Globals.k)+suffix)
	V = Vt.T
	# normalize
	for i in range(Globals.nItems):
		V[i] /= np.linalg.norm(V[i])

	for i in range(Globals.step,Globals.nUsers):
		known = data[i]!=0
		y = data[i,known]
		X = V[known]

		clf = KernelRidge(alpha=a,kernel=kernel)
		clf.fit(X, y)
		pred = clf.predict(V)
		A[i] = pred
		# mask = test[i]!=0
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

def predictionWithCombi():
	A1 = np.load('./log/KRR_A_'+str(Globals.k)+'.npy')
	A2 = np.load('./log/KRR_A_'+str(Globals.k)+'_2.npy')
	A3 = np.load('./log/KRR_A_'+str(Globals.k)+'_3.npy')
	A = (A1+A2+A3)/3.0
	return A

if __name__ == '__main__':
	Initialization.initialization()
	if Globals.fixed:
		data, test = Initialization.readInData2(idx=Globals.dataIdx)
		if Globals.predict == 'a':
			chooseAlpha(data,test)
		else:
			A = KRR(data,test)
			np.save('./log/KRR'+Globals.modelIdx+'_A_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',A)
	else:
		data = Initialization.readInData('./data/data_train.csv')
		data, test = SVD.splitData(data,10)
		if Globals.predict=='c':
			A = predictionWithCombi()
			np.save('./log/KRR_A_'+str(Globals.k)+'_combi.npy',A)
		else:
			A = KRR(data,test)
			np.save('./log/KRR_A_'+str(Globals.k)+Globals.modelIdx+'.npy',A)
		SVD.writeOutData(A)
