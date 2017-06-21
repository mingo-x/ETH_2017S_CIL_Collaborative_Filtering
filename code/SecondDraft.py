# continuous rating for prediction
import numpy as np
import csv
from sys import argv
import time

# globals
global k, outputIdx
k = 1000
outputIdx = ''

nUsers = 10000
nItems = 1000

def initialization():
	global k, outputIdx
	for i in range(1,len(argv)):
		if argv[i].startswith('-k='):
			k = int(argv[i][3:])
		elif argv[i].startswith('-o='):
			outputIdx = argv[i][3:]

def readInData(inPath):
# read in data
	print('start reading data')
	startTime = time.time()
	data = np.zeros((nUsers,nItems))
	csvReader = csv.reader(open(inPath,encoding='utf-8'))
	abort = True
	for row in csvReader:
		if abort:
			abort = False
			continue
		idx = row[0]
		val = int(row[1])
		npos = idx.index('_')
		i = int(idx[1:npos])-1
		j = int(idx[npos+2:])-1
		data[i,j] = val
	endTime = time.time()
	print('finish reading data', int(endTime-startTime), 's')
	return data

def fillInMissing(data):
# fill in missing data
	print('start filling in missing data')
	for i in range(nItems):
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
	S = np.zeros((nUsers, nItems))
	S[:nUsers, :nItems] = np.diag(s)
	return U, S, Vt

def prediction(U,S,Vt):
	print('start matrix multiplication')
	Sk = S[:k,:k]
	Ak = U[:,:k].dot(Sk).dot(Vt[:k,:])
	print('finish matrix multiplication')
	return Ak

def writeOutData(samplePath = './data/sampleSubmission.csv'):
# write prediction
	print('start writing data')
	startTime = time.time()
	csvReader = csv.reader(open(samplePath,encoding='utf-8'))
	csvWriter = csv.writer(open('./data/prediction'+outputIdx+'.csv','w',newline=''))
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

if __name__ == "__main__":
	initialization()
	print('k =',k)
	print('output idx =',outputIdx)
	data = readInData('./data/data_train.csv')
	fillInMissing(data)
	U, S, Vt = SVD(data)
	Ak = prediction(U, S, Vt)
	writeOutData()

