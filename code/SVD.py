# continuous rating for prediction
import csv
from sys import argv
import time
import random
import Initialization
import Globals
import numpy as np

def splitData(data, n = 10):
	print('start splitting data')
	p = 1.0/n
	trainMask = data!=0
	testMask = trainMask.copy()
	for i in range(Globals.nUsers):
		for j in range(Globals.nItems):
			if trainMask[i,j]:
				r = random.random()
				if r > p:
					testMask[i,j] = False
				else:
					trainMask[i,j] = False
	train = data.copy()
	test = data.copy()
	train[testMask] = 0
	test[trainMask] = 0
	print('finish splitting data train num:',np.count_nonzero(train),'test num:', np.sum(testMask))
	return train, test

def splitData2(data, n = 10,idx=''):
	print('start splitting data')
	p = 1.0/n
	trainMask = data!=0
	testMask = trainMask.copy()
	for i in range(Globals.nUsers):
		for j in range(Globals.nItems):
			if trainMask[i,j]:
				r = random.random()
				if r > p:
					testMask[i,j] = False
				else:
					trainMask[i,j] = False
	train = data.copy()
	test = data.copy()
	train[testMask] = 0
	test[trainMask] = 0
	print('finish splitting data train num:',np.count_nonzero(train),'test num:', np.count_nonzero(test))
	np.save('./data/train'+idx+'.npy',train)
	np.save('./data/test'+idx+'.npy',test)

def fillInMissing(data):
# fill in missing data
	print('start filling in missing data')
	for i in range(Globals.nItems):
		missing = data[:,i]==0
		known = missing==False
		mean_of_known = np.mean(data[known,i])
		data[missing,i] = mean_of_known
	print('finish filling in missing data')

def SVD(data):
# SVD
	print('start SVD')
	startTime = time.time()
	U, s, Vt = np.linalg.svd(data, full_matrices=True)
	endTime = time.time()
	print('finish SVD', U.shape, s.shape, Vt.shape, int(endTime-startTime), 's')
	# print(s)
	S = np.zeros((Globals.nUsers, Globals.nItems))
	S[:min(Globals.nUsers,Globals.nItems), :min(Globals.nUsers,Globals.nItems)] = np.diag(s)
	return U, S, Vt

def prediction(U,S,Vt,k):
	print('start matrix multiplication k =',k)
	Sk = S[:k,:k]
	Ak = U[:,:k].dot(Sk).dot(Vt[:k,:])
	print('finish matrix multiplication')
	return Ak

def predictionWithClipping(U,S,Vt,k):
	print('start matrix multiplication k =',k)
	Sk = S[:k,:k]
	Ak = U[:,:k].dot(Sk).dot(Vt[:k,:])
	# over 5
	mask = Ak>5
	Ak[mask] = 5
	# below 1
	mask = Ak<1
	Ak[mask] = 1
	print('finish matrix multiplication')
	return Ak

def predictionWithClippingByStep(U,S,Vt,k):
	print('start matrix multiplication k =',k)
	Sk = S[:k,:k]
	Ak = np.zeros((Globals.nUsers,Globals.nItems))
	for i in range(k):
		Tk = U[:, i:i+1].dot(Sk[i:i+1,i:i+1]).dot(Vt[i:i+1,:])
		Ak += Tk
		# over 5
		mask = Ak>5
		Ak[mask] = 5
		# below 1
		mask = Ak<1
		Ak[mask] = 1
	print('finish matrix multiplication')
	return Ak

def evaluation(data,Ak,testMask):
	sumSquare = np.sum(np.square((Ak-data)[testMask]))
	res = np.sqrt(sumSquare/np.count_nonzero(testMask))
	return res

def evaluation2(A,test):
	mask = test!=0
	res = np.sqrt(np.mean(np.square((A-test)[mask])))
	return res

def writeOutData(Ak,samplePath = './data/sampleSubmission.csv'):
# write prediction
	print('start writing data')
	startTime = time.time()
	csvReader = csv.reader(open(samplePath,encoding='utf-8'))
	csvWriter = csv.writer(open('./data/prediction'+Globals.outputIdx+'.csv','w',newline=''))
	abort = True
	for row in csvReader:
		if abort:
			csvWriter.writerow(row)
			abort = False
			continue
		idx = row[0]
		npos = idx.index('_')
		i = int(idx[1:npos])-1
		j = int(idx[npos+2:])-1
		csvWriter.writerow([idx,Ak[i,j]])
	endTime = time.time()
	print('finish writing data', int(endTime-startTime), 's')

def chooseK(data,n=10):
	csvWriter = csv.writer(open('./log/k.csv','w',newline=''))
	csvWriter.writerow(['k','RMSE'])
	for k in range(440,1001,20):
		startTime = time.time()
		print('k =',k)
		scores = []
		for i in range(n):
			print('round',i+1)
			train, testMask = splitData(data,n)
			fillInMissing(train)
			U, S, Vt = SVD(train)
			Ak = prediction(U, S, Vt, k)
			score = evaluation(data,Ak,testMask)
			print(i+1,'RMSE =',score)
			scores.append(score)
		ave = np.mean(scores)
		endTime = time.time()
		print('k =',k, 'average RMSE =', ave, int(endTime-startTime), 's')
		csvWriter.writerow([k,ave])

if __name__ == "__main__":
	Initialization.initialization()
	data = Initialization.readInData('./data/data_train.csv')
	# chooseK(data)
	splitData2(data,idx='1')

