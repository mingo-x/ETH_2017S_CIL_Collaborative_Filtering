# DESCRIPTION: This is the file to combine all the models into one ensembled model. For each training/validation data split, we fit a ridge regression on the validation data and obtain a weighted average of the predictions from different models. Features include the predictions of all the models and the two-way interactions between some models. The final prediction is an average of the ensembled results of different training/validation data splits.

# USAGE: To train the ensemble for one data split, run "python3 code/Ensemble.py -p=tr" and "python3 code/Ensemble.py -p=tr -d=1". "-p=tr" stands for adding two-way interaction and using ridge regression. "-d" is the option for choosing data split. To get the final prediction, run "python3 code/Ensemble.py -p=a -d=tr". "-p=a" means taking average of different ensembles and "-d=tr" specifies the ensemble method.

from sklearn import linear_model
import SVD
import numpy as np
import Globals
import time
import Initialization

def loadData(data,predMask):
	print('start initialization')
	startTime = time.time()
	known = data!=0
	nObs = np.count_nonzero(data)
	target = np.reshape(data[known],(nObs,1))

	# load models
	Basic1_A = np.load('./log/Basic1_A_fixed'+Globals.dataIdx+'.npy')
	Basic2_A = np.load('./log/Basic2_A_fixed'+Globals.dataIdx+'.npy')
	Basic3_A = np.load('./log/Basic3_A_fixed'+Globals.dataIdx+'.npy')
	Basic4_A = np.load('./log/Basic4_A_fixed'+Globals.dataIdx+'.npy')
	Basic5_A = np.load('./log/Basic5_A_fixed'+Globals.dataIdx+'.npy')
	Basic6_A = np.load('./log/Basic6_A_fixed'+Globals.dataIdx+'.npy')
	KMeans_A = np.load('./log/Kmeans_A_combi_fixed'+Globals.dataIdx+'.npy')
	PSVD_A = np.load('./log/PSVD_A_12_fixed'+Globals.dataIdx+'.npy')
	LM_A = np.load('./log/LM_A_fixed'+Globals.dataIdx+'.npy')
	# GRSVD_A = np.load('./log/GRSVD_A_32_fixed'+Globals.dataIdx+'.npy')
	RSVDF_A = np.load('./log/RSVDF_A_32_fixed'+Globals.dataIdx+'.npy')
	RSVDF2_A = np.load('./log/RSVDF2_A_32_fixed'+Globals.dataIdx+'.npy')
	KRR_A = np.load('./log/KRR_A_32_fixed'+Globals.dataIdx+'.npy')
	KRR2_A = np.load('./log/KRR2_A_32_fixed'+Globals.dataIdx+'.npy')
	UB_A = np.load('./log/UB_A_500_fixed'+Globals.dataIdx+'.npy')

	# load training data
	train = np.append([Basic1_A[known]],[Basic2_A[known]],axis=0)
	train = np.append(train,[Basic3_A[known]],axis=0)
	train = np.append(train,[Basic4_A[known]],axis=0)
	train = np.append(train,[Basic5_A[known]],axis=0)
	train = np.append(train,[Basic6_A[known]],axis=0)
	train = np.append(train,[KMeans_A[known]],axis=0)
	train = np.append(train,[PSVD_A[known]],axis=0)
	train = np.append(train,[RSVDF_A[known]],axis=0)
	train = np.append(train,[RSVDF2_A[known]],axis=0)
	train = np.append(train,[KRR_A[known]],axis=0)
	train = np.append(train,[KRR2_A[known]],axis=0)
	train = np.append(train,[LM_A[known]],axis=0)
	train = np.append(train,[UB_A[known]],axis=0)
	# train = np.append(train,[GRSVD_A[known]],axis=0)
	#train = np.append(train,[NSVD2_A[known]],axis=0)

	# two-way interaction
	if Globals.predict == 't' or Globals.predict=='tr': 
		# train = np.append(train,[np.multiply(GRSVD_A[known],PSVD_A[known])],axis=0)
		# train = np.append(train,[np.multiply(GRSVD_A[known],RSVDF_A[known])],axis=0)
		# train = np.append(train,[np.multiply(GRSVD_A[known],RSVDF2_A[known])],axis=0)
		# train = np.append(train,[np.multiply(GRSVD_A[known],KRR2_A[known])],axis=0)
		# train = np.append(train,[np.multiply(GRSVD_A[known],LM_A[known])],axis=0)
		train = np.append(train,[np.multiply(PSVD_A[known],RSVDF_A[known])],axis=0)
		train = np.append(train,[np.multiply(PSVD_A[known],RSVDF2_A[known])],axis=0)
		train = np.append(train,[np.multiply(PSVD_A[known],KRR_A[known])],axis=0)
		train = np.append(train,[np.multiply(PSVD_A[known],KRR2_A[known])],axis=0)
		train = np.append(train,[np.multiply(PSVD_A[known],LM_A[known])],axis=0)
		train = np.append(train,[np.multiply(PSVD_A[known],UB_A[known])],axis=0)
		# train = np.append(train,[np.multiply(RSVDF_A[known],RSVDF2_A[known])],axis=0)
		# train = np.append(train,[np.multiply(RSVDF_A[known],KRR_A[known])],axis=0)
		# train = np.append(train,[np.multiply(RSVDF_A[known],KRR2_A[known])],axis=0)
		# train = np.append(train,[np.multiply(RSVDF_A[known],LM_A[known])],axis=0)
		# train = np.append(train,[np.multiply(RSVDF_A[known],UB_A[known])],axis=0)
		train = np.append(train,[np.multiply(RSVDF2_A[known],KRR_A[known])],axis=0)
		train = np.append(train,[np.multiply(RSVDF2_A[known],KRR2_A[known])],axis=0)
		train = np.append(train,[np.multiply(RSVDF2_A[known],LM_A[known])],axis=0)
		train = np.append(train,[np.multiply(RSVDF2_A[known],UB_A[known])],axis=0)
		train = np.append(train,[np.multiply(KRR_A[known],KRR2_A[known])],axis=0)
		train = np.append(train,[np.multiply(KRR_A[known],LM_A[known])],axis=0)
		train = np.append(train,[np.multiply(KRR_A[known],UB_A[known])],axis=0)
		# train = np.append(train,[np.multiply(KRR2_A[known],LM_A[known])],axis=0)
		# train = np.append(train,[np.multiply(KRR2_A[known],UB_A[known])],axis=0)
		train = np.append(train,[np.multiply(LM_A[known],UB_A[known])],axis=0)
	train = train.T
	print(train.shape)

	# load prediction data
	test = np.append([Basic1_A[predMask]],[Basic2_A[predMask]],axis=0)
	Basic1_A = None
	Basic2_A = None
	test = np.append(test,[Basic3_A[predMask]],axis=0)
	Basic3_A = None
	test = np.append(test,[Basic4_A[predMask]],axis=0)
	Basic4_A = None
	test = np.append(test,[Basic5_A[predMask]],axis=0)
	Basic5_A = None
	test = np.append(test,[Basic6_A[predMask]],axis=0)
	Basic6_A = None
	test = np.append(test,[KMeans_A[predMask]],axis=0)
	KMeans_A = None
	test = np.append(test,[PSVD_A[predMask]],axis=0)
	test = np.append(test,[RSVDF_A[predMask]],axis=0)
	test = np.append(test,[RSVDF2_A[predMask]],axis=0)
	test = np.append(test,[KRR_A[predMask]],axis=0)
	test = np.append(test,[KRR2_A[predMask]],axis=0)
	test = np.append(test,[LM_A[predMask]],axis=0)
	test = np.append(test,[UB_A[predMask]],axis=0)
	# test = np.append(test,[GRSVD_A[predMask]],axis=0)
	#test = np.append(test,[NSVD2_A[predMask]],axis=0)

	# two-way interaction
	if Globals.predict == 't' or Globals.predict=='tr':
		# test = np.append(test,[np.multiply(GRSVD_A[predMask],PSVD_A[predMask])],axis=0)
		# test = np.append(test,[np.multiply(GRSVD_A[predMask],RSVDF_A[predMask])],axis=0)
		# test = np.append(test,[np.multiply(GRSVD_A[predMask],RSVDF2_A[predMask])],axis=0)
		# test = np.append(test,[np.multiply(GRSVD_A[predMask],KRR2_A[predMask])],axis=0)
		# test = np.append(test,[np.multiply(GRSVD_A[predMask],LM_A[predMask])],axis=0)
		test = np.append(test,[np.multiply(PSVD_A[predMask],RSVDF_A[predMask])],axis=0)
		test = np.append(test,[np.multiply(PSVD_A[predMask],RSVDF2_A[predMask])],axis=0)
		test = np.append(test,[np.multiply(PSVD_A[predMask],KRR_A[predMask])],axis=0)
		test = np.append(test,[np.multiply(PSVD_A[predMask],KRR2_A[predMask])],axis=0)
		test = np.append(test,[np.multiply(PSVD_A[predMask],LM_A[predMask])],axis=0)
		test = np.append(test,[np.multiply(PSVD_A[predMask],UB_A[predMask])],axis=0)
		PSVD_A = None
		# test = np.append(test,[np.multiply(RSVDF_A[predMask],RSVDF2_A[predMask])],axis=0)
		# test = np.append(test,[np.multiply(RSVDF_A[predMask],KRR_A[predMask])],axis=0)
		# test = np.append(test,[np.multiply(RSVDF_A[predMask],KRR2_A[predMask])],axis=0)
		# test = np.append(test,[np.multiply(RSVDF_A[predMask],LM_A[predMask])],axis=0)
		# test = np.append(test,[np.multiply(RSVDF_A[predMask],UB_A[predMask])],axis=0)
		RSVDF_A = None
		test = np.append(test,[np.multiply(RSVDF2_A[predMask],KRR_A[predMask])],axis=0)
		test = np.append(test,[np.multiply(RSVDF2_A[predMask],KRR2_A[predMask])],axis=0)
		test = np.append(test,[np.multiply(RSVDF2_A[predMask],LM_A[predMask])],axis=0)
		test = np.append(test,[np.multiply(RSVDF2_A[predMask],UB_A[predMask])],axis=0)
		RSVDF2_A = None
		test = np.append(test,[np.multiply(KRR_A[predMask],KRR2_A[predMask])],axis=0)
		test = np.append(test,[np.multiply(KRR_A[predMask],LM_A[predMask])],axis=0)
		test = np.append(test,[np.multiply(KRR_A[predMask],UB_A[predMask])],axis=0)
		KRR_A = None
		# test = np.append(test,[np.multiply(KRR2_A[predMask],LM_A[predMask])],axis=0)
		# test = np.append(test,[np.multiply(KRR2_A[predMask],UB_A[predMask])],axis=0)
		KRR2_A = None
		test = np.append(test,[np.multiply(LM_A[predMask],UB_A[predMask])],axis=0)
		LM_A = None
		UB_A = None
	else:
		PSVD_A = None
		RSVDF_A = None
		RSVDF2_A = None
		KRR_A = None
		KRR2_A = None
		LM_A = None
		UB_A = None
	test = test.T
	print(test.shape)
	
	endTime = time.time()
	print('finish initialization',int(endTime-startTime),'s')
	return train, target, test

# linear regression
def ensemble(train, target, test, predMask):
	print('start training')
	startTime = time.time()
	regr = linear_model.LinearRegression()
	regr.fit(train, target)
	endTime = time.time()
	print('finish training',int(endTime-startTime),'s')
	print('Coefficients: \n', regr.coef_)

	print('start predicting')
	startTime = time.time()
	pred = regr.predict(test)
	c = np.count_nonzero(predMask)
	pred = np.reshape(pred,c)
	endTime = time.time()
	print('finish predicting',int(endTime-startTime),'s',pred.shape)
	A = np.empty((Globals.nUsers,Globals.nItems))
	A[predMask] = pred
	return A

# ridge regression
def ensembleRR(train, target, test, predMask):
	print('start ridge regression')
	startTime = time.time()
	regr = linear_model.Lasso(alpha=0.1)
	# regr = linear_model.Ridge(alpha=1.0, tol=1e-5)
	regr.fit(train, target)
	endTime = time.time()
	print('finish training',int(endTime-startTime),'s')
	print('Coefficients: \n', regr.coef_)

	print('start predicting')
	startTime = time.time()
	pred = regr.predict(test)
	c = np.count_nonzero(predMask)
	pred = np.reshape(pred,c)
	endTime = time.time()
	print('finish predicting',int(endTime-startTime),'s',pred.shape)
	A = np.empty((Globals.nUsers,Globals.nItems))
	A[predMask] = pred

	return A

# average of different ensembles
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
	# average ensembles
	if Globals.predict == 'a':
		A = average()
		np.save('./log/Ensemble_A_ave.npy',A)
		SVD.writeOutData(A)
	else:
		train, data = Initialization.readInData2(idx = Globals.dataIdx)
		predMask = Initialization.readInSubmission2('./data/sampleSubmission.csv')
		train, target, test = loadData(data,predMask)
		# ridge regression, without two-way interaction
		if Globals.predict == 'r':
			A = ensembleRR(train,target,test,predMask)
			np.save('./log/Ensemble_A'+Globals.dataIdx+'_r.npy',A)
		# ridge regression, with two-way interaction
		elif Globals.predict == 'tr':
			A = ensembleRR(train,target,test,predMask)
			np.save('./log/Ensemble_A'+Globals.dataIdx+'_tr.npy',A)
		# linear regression, with two-way interaction
		elif Globals.predict == 't':
			A = ensemble(train,target,test,predMask)
			np.save('./log/Ensemble_A'+Globals.dataIdx+'_t.npy',A)
		# linear regression, without two-way interaction
		else:
			A = ensemble(train,target,test,predMask)
			np.save('./log/Ensemble_A'+Globals.dataIdx+'.npy',A)