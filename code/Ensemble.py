from sklearn import linear_model
import SVD
import numpy as np
import Globals
import time
import Initialization

def loadData(data):
	print('start initialization')
	startTime = time.time()
	known = data!=0
	nObs = np.count_nonzero(data)
	target = np.reshape(data[known],(nObs,1))

	Basic1_A = np.load('./log/Basic1_A_fixed'+Globals.dataIdx+'.npy')
	Basic2_A = np.load('./log/Basic2_A_fixed'+Globals.dataIdx+'.npy')
	Basic3_A = np.load('./log/Basic3_A_fixed'+Globals.dataIdx+'.npy')
	Basic4_A = np.load('./log/Basic4_A_fixed'+Globals.dataIdx+'.npy')
	Basic5_A = np.load('./log/Basic5_A_fixed'+Globals.dataIdx+'.npy')
	Basic6_A = np.load('./log/Basic6_A_fixed'+Globals.dataIdx+'.npy')
	LM_A = np.load('./log/LM_A_fixed'+Globals.dataIdx+'.npy')
	if Globals.dataIdx == '1':
		KMeans_A = np.load('./log/Kmeans_A_combi_fixed'+Globals.dataIdx+'_2.npy')
		PSVD_A = np.load('./log/PSVD_A_12_fixed'+Globals.dataIdx+'.npy')
		RSVD_A = np.load('./log/RSVD_A_10_fixed'+Globals.dataIdx+'.npy')
		RSVD2_A = np.load('./log/RSVD2_A_5_fixed'+Globals.dataIdx+'.npy')
		KRR_A = np.load('./log/KRR_A_15_fixed'+Globals.dataIdx+'.npy')
	else:
		KMeans_A = np.load('./log/Kmeans_A_combi_fixed'+Globals.dataIdx+'.npy')
		PSVD_A = np.load('./log/PSVD_A_20_fixed'+Globals.dataIdx+'.npy')
		RSVD_A = np.load('./log/RSVD_A_20_fixed'+Globals.dataIdx+'.npy')
		RSVD2_A = np.load('./log/RSVD2_A_20_fixed'+Globals.dataIdx+'.npy')
		KRR_A = np.load('./log/KRR_A_20_fixed'+Globals.dataIdx+'.npy')
	
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
	if Globals.predict == 't' or Globals.predict=='tr': # two-way interaction
		train = np.append(train,[np.multiply(PSVD_A[known],RSVD_A[known])],axis=0)
		train = np.append(train,[np.multiply(PSVD_A[known],RSVD2_A[known])],axis=0)
		train = np.append(train,[np.multiply(PSVD_A[known],KRR_A[known])],axis=0)
		train = np.append(train,[np.multiply(PSVD_A[known],LM_A[known])],axis=0)
		train = np.append(train,[np.multiply(RSVD_A[known],RSVD2_A[known])],axis=0)
		train = np.append(train,[np.multiply(RSVD_A[known],KRR_A[known])],axis=0)
		train = np.append(train,[np.multiply(RSVD_A[known],LM_A[known])],axis=0)
		train = np.append(train,[np.multiply(RSVD2_A[known],KRR_A[known])],axis=0)
		train = np.append(train,[np.multiply(RSVD2_A[known],LM_A[known])],axis=0)
	train = train.T

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
	if Globals.predict == 't' or Globals.predict=='tr':
		test = np.append(test,[np.multiply(PSVD_A.flatten(),RSVD_A.flatten())],axis=0)
		test = np.append(test,[np.multiply(PSVD_A.flatten(),RSVD2_A.flatten())],axis=0)
		test = np.append(test,[np.multiply(PSVD_A.flatten(),KRR_A.flatten())],axis=0)
		test = np.append(test,[np.multiply(PSVD_A.flatten(),LM_A.flatten())],axis=0)
		test = np.append(test,[np.multiply(RSVD_A.flatten(),RSVD2_A.flatten())],axis=0)
		test = np.append(test,[np.multiply(RSVD_A.flatten(),KRR_A.flatten())],axis=0)
		test = np.append(test,[np.multiply(RSVD_A.flatten(),LM_A.flatten())],axis=0)
		test = np.append(test,[np.multiply(RSVD2_A.flatten(),KRR_A.flatten())],axis=0)
		test = np.append(test,[np.multiply(RSVD2_A.flatten(),LM_A.flatten())],axis=0)
	test = test.T
	endTime = time.time()
	print('finish initialization',int(endTime-startTime),'s',train.shape, test.shape)
	return train, target, test

def ensemble(train, target, test):
	print('start training')
	startTime = time.time()
	regr = linear_model.LinearRegression()
	regr.fit(train, target)
	endTime = time.time()
	print('finish training',int(endTime-startTime),'s')
	print('Coefficients: \n', regr.coef_)

	print('start predicting')
	startTime = time.time()
	A = regr.predict(test)
	endTime = time.time()
	print('finish predicting',int(endTime-startTime),'s',A.shape)
	A = np.reshape(A,(Globals.nUsers,Globals.nItems))

	return A

def ensembleRR(train, target, test):
	print('start ridge regression')
	startTime = time.time()
	regr = linear_model.Ridge(alpha=0.5, tol=1e-4)
	regr.fit(train, target)
	endTime = time.time()
	print('finish training',int(endTime-startTime),'s')
	print('Coefficients: \n', regr.coef_)

	print('start predicting')
	startTime = time.time()
	A = regr.predict(test)
	endTime = time.time()
	print('finish predicting',int(endTime-startTime),'s',A.shape)
	A = np.reshape(A,(Globals.nUsers,Globals.nItems))

	return A

def average():
	suffix = '.npy'
	if Globals.dataIdx!= '':
		suffix = '_'+Globals.dataIdx+'.npy'
	A0 = np.load('./log/Ensemble_A'+suffix)
	A1 = np.load('./log/Ensemble_A1'+suffix)
	A = (A0+A1)/2.0
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
	if Globals.predict == 'a':
		A = average()
		np.save('./log/Ensemble_A_ave.npy',A)
	else:
		train, data = Initialization.readInData2(idx = Globals.dataIdx)
		# data = Initialization.readInData('./data/data_train.csv')
		train, target, test = loadData(data)
		if Globals.predict == 'r':
			A = ensembleRR(train,target,test)
			np.save('./log/Ensemble_A'+Globals.dataIdx+'_r.npy',A)
		elif Globals.predict == 'tr':
			A = ensembleRR(train,target,test)
			np.save('./log/Ensemble_A'+Globals.dataIdx+'_tr.npy',A)
		elif Globals.predict == 't':
			A = ensemble(train,target,test)
			np.save('./log/Ensemble_A'+Globals.dataIdx+'_t.npy',A)
		else:
			A = ensemble(train,target,test)
			np.save('./log/Ensemble_A'+Globals.dataIdx+'.npy',A)
	SVD.writeOutData(A)