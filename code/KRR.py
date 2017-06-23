import Globals
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import Initialization
import SVD

def kernel(x1,x2):
	return np.exp(2*(np.dot(x1,x2)-1))

def KRR(data):
	if Globals.step == 0:
		A = data.copy()
	else:
		A = np.load('./log/KRR_A_'+str(Globals.k)+Globals.modelIdx+'_'+Globals.step+'.npy')
	Vt = np.load('./log/RSVD_Vt_'+str(Globals.k)+Globals.modelIdx+'.npy')
	V = Vt.T
	# normalize
	for i in range(Globals.nItems):
		V[i] /= np.linalg.norm(V[i])
	for i in range(Globals.nUsers):
		print('user ',i+1)
		known = data[i]!=0
		missing = known==False
		y = data[i,known]
		X = V[known]

		clf = KernelRidge(alpha=0.5,kernel=kernel)
		clf.fit(X, y)
		pred = clf.predict(V[missing])
		#clipping
		mask = pred>5
		pred[mask] = 5
		mask = pred<1
		pred[mask] = 1
		A[i,missing] = pred

		if i%1000 == 0:
			np.save('./log/KRR_A_'+str(Globals.k)+Globals.modelIdx+'_'+i+'.npy',A)

	known = data!=0
	return A

def predictionWithCombi():
	A1 = np.load('./log/KRR_A_'+str(k)+'.npy')
	A2 = np.load('./log/KRR_A_'+str(k)+'_2.npy')
	A3 = np.load('./log/KRR_A_'+str(k)+'_3.npy')
	A = (A1+A2+A3)/3.0
	return A

if __name__ == '__main__':
	Initialization.initialization()
	data = Initialization.readInData('./data/data_train.csv')
	if Globals.predict=='c':
		A = predictionWithCombi()
		np.save('./log/KRR_A_'+str(Globals.k)+'_combi.npy',A)
	else:
		A = KRR(data)
		np.save('./log/KRR_A_'+str(Globals.k)+Globals.modelIdx+'.npy',A)
	SVD.writeOutData(A)
