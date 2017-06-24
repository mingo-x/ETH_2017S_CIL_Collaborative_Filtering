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

def KRR(data,test):
	suffix = '_fixed'+Globals.dataIdx+'.npy'
	if not Globals.fixed:
		suffix = '.npy'
	if Globals.step == 0:
		A = data.copy()
	else:
		A = np.load('./log/KRR_A_'+str(Globals.k)+Globals.modelIdx+'_'+str(Globals.step)+suffix)
	Vt = np.load('./log/RSVD_Vt_'+str(Globals.k)+Globals.modelIdx+suffix)
	V = Vt.T
	# normalize
	for i in range(Globals.nItems):
		V[i] /= np.linalg.norm(V[i])
	for i in range(Globals.step,Globals.nUsers):
		print('user ',i+1,)
		known = data[i]!=0
		y = data[i,known]
		X = V[known]

		clf = KernelRidge(alpha=0.5,kernel=kernel)
		clf.fit(X, y)
		pred = clf.predict(V)
		A[i] = pred
		mask = test[i]!=0
		# score = np.sqrt(np.mean(np.square(pred[mask]-test[i,mask])))
		# print('score =',score)

		if i%1000 == 0:
			np.save('./log/KRR_A_'+str(Globals.k)+Globals.modelIdx+'_'+str(i)+suffix,A)

	score = SVD.evaluation2(A,test)
	print('test error =',score)
	#clipping
	mask = A>5
	A[mask] = 5
	mask = A<1
	A[mask] = 1
	score = SVD.evaluation2(A,test)
	print('after clipping test error =',score)
	return A

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
		# topRatedMovies(data)
		# return
		A = KRR(data,test)
		np.save('./log/KRR_A_'+str(Globals.k)+'_fixed1.npy',A)
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
