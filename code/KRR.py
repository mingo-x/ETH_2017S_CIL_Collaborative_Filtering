import Globals
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import Initialization

def kernel(x1,x2):
	return np.exp(2*(np.dot(x1,x2)-1))

def KRR(data):
	A = data.copy()
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

	known = data!=0
	trainigErr = np.sqrt(np.mean(np.square((data-A)[known])))
	print('training error =',trainigErr)
	return A

if __name__ == '__main__':
	Initialization.initialization()
	data = Initialization.readInData('./data/data_train.csv')
	A = KRR(data)
	np.save('./log/KRR_A_'+str(Globals.k)+Globals.modelIdx+'.npy',A)
	SVD.writeOutData(A)
