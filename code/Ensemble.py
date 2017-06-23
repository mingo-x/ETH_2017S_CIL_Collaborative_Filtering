from sklearn import linear_model
import SVD
import numpy as np
import Globals
import time
import Initialization

def ensemble(data):
	print('start initialization')
	startTime = time.time()
	known = data!=0
	nObs = np.count_nonzero(data)
	target = np.reshape(data[known],(nObs,1))

	Basic1_A = np.load('./log/Basic1_A_fixed.npy')
	Basic2_A = np.load('./log/Basic2_A_fixed.npy')
	Basic3_A = np.load('./log/Basic3_A_fixed.npy')
	Basic4_A = np.load('./log/Basic4_A_fixed.npy')
	Basic5_A = np.load('./log/Basic5_A_fixed.npy')
	Basic6_A = np.load('./log/Basic6_A_fixed.npy')
	KMeans_A = np.load('./log/Kmeans_A_combi_fixed.npy')
	PSVD_A = np.load('./log/PSVD_A_20_fixed.npy')
	RSVD_A = np.load('./log/RSVD_A_20_fixed.npy')
	RSVD2_A = np.load('./log/RSVD2_A_20_fixed.npy')
	KRR_A = np.load('./log/KRR_A_20_fixed.npy')
	LM_A = np.load('./log/LM_A_fixed.npy')
	#NSVD2_A = np.load('./log/NSVD_A_20_fixed.npy')

	train = np.append([Basic1_A[known]],[Basic2_A[known]],axis=0)
	train = np.append(train,[Basic3_A[known]],axis=0)
	train = np.append(train,[Basic4_A[known]],axis=0)
	train = np.append(train,[Basic5_A[known]],axis=0)
	train = np.append(train,[Basic6_A[known]],axis=0)
	train = np.append(train,[KMeans_A[known]],axis=0)
	train = np.append(train,[PSVD_A[known]],axis=0)
	train = np.append(train,[RSVD_A[known]],axis=0)
	train = np.append(train,[RSVD2_A[known]],axis=0)
	train = np.append(train,[KRR_A[known]],axis=0)
	train = np.append(train,[LM_A[known]],axis=0)
	#train = np.append(train,[NSVD2_A[known]],axis=0)
	train = train.T
	endTime = time.time()
	print('finish initialization',int(endTime-startTime),'s',train.shape)

	print('start training')
	startTime = time.time()
	regr = linear_model.LinearRegression()
	regr.fit(train, target)
	endTime = time.time()
	print('finish training',int(endTime-startTime),'s')
	print('Coefficients: \n', regr.coef_)

	print('start predicting')
	startTime = time.time()
	test = np.append([Basic1_A.flatten()],[Basic2_A.flatten()],axis=0)
	test = np.append(test,[Basic3_A.flatten()],axis=0)
	test = np.append(test,[Basic4_A.flatten()],axis=0)
	test = np.append(test,[Basic5_A.flatten()],axis=0)
	test = np.append(test,[Basic6_A.flatten()],axis=0)
	test = np.append(test,[KMeans_A.flatten()],axis=0)
	test = np.append(test,[PSVD_A.flatten()],axis=0)
	test = np.append(test,[RSVD_A.flatten()],axis=0)
	test = np.append(test,[RSVD2_A.flatten()],axis=0)
	test = np.append(test,[KRR_A.flatten()],axis=0)
	test = np.append(test,[LM_A.flatten()],axis=0)
	#test = np.append(test,[NSVD2_A.flatten()],axis=0)
	test = test.T

	A = regr.predict(test)
	endTime = time.time()
	print('finish predicting',int(endTime-startTime),'s',A.shape)
	A = np.reshape(A,(Globals.nUsers,Globals.nItems))

	#clipping
	# over 5
	mask = A>5
	A[mask] = 5
	# below 1
	mask = A<1
	A[mask] = 1
	return A

if __name__ == "__main__":
	Initialization.initialization()
	train, data = Initialization.readInData2()
	A = ensemble(data)
	np.save('./log/Ensemble_A.npy',A)
	SVD.writeOutData(A)