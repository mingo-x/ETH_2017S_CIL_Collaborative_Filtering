# DESCRIPTION: This file defines some helper functions for parameter initialization and data reading

import numpy as np
import csv
from sys import argv
import time
import Globals

# parameter initialization
def initialization():
	for i in range(1,len(argv)):
		if argv[i].startswith('-k='):
			Globals.k = int(argv[i][3:])
			print('k =',Globals.k)
		elif argv[i].startswith('-o='):
			Globals.outputIdx = argv[i][3:]
			print('output idx =', Globals.outputIdx)
		elif argv[i].startswith('-w='):
			Globals.warmStart = argv[i][3] == 't'
			print('warm start =', Globals.warmStart)
		elif argv[i].startswith('-l='):
			Globals.lrate = float(argv[i][3:])
			print('learning rate =', Globals.lrate)
		elif argv[i].startswith('-p='):
			Globals.predict = argv[i][3:]
			print('predict =', Globals.predict)
		elif argv[i].startswith('-i='):
			Globals.modelIdx = argv[i][3:]
			print('model idx =', Globals.modelIdx)
		elif argv[i].startswith('-s='):
			Globals.step = int(argv[i][3:])
			print('starting step =', Globals.step)
		elif argv[i].startswith('-f='):
			Globals.fixed = argv[i][3] == 't'
			print('fixed split =', Globals.fixed)
		elif argv[i].startswith('-d='):
			Globals.dataIdx = argv[i][3:]
			print('data idx =', Globals.dataIdx)
		elif argv[i].startswith('-l2='):
			Globals.l2 = float(argv[i][4:])
			print('lambda2 =',Globals.l2)

# read in raw data
def readInData(inPath):
	print('start reading data')
	startTime = time.time()
	data = np.zeros((Globals.nUsers,Globals.nItems))
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

# read in splitted data
def readInData2(trainPath='./data/train',testPath='./data/test',idx=''):
	print('start reading data')
	startTime = time.time()
	print('train =',trainPath+idx+'.npy')
	print('test =',testPath+idx+'.npy')
	train = np.load(trainPath+idx+'.npy')
	test = np.load(testPath+idx+'.npy')
	endTime = time.time()
	print('finish reading data', int(endTime-startTime), 's', train.shape, test.shape)
	return train,test

# read in submission indices and return a list
def readInSubmission(inPath):
	print('start reading submission data')
	startTime = time.time()
	index = []
	csvReader = csv.reader(open(inPath,encoding='utf-8'))
	abort = True
	for row in csvReader:
		if abort:
			abort = False
			continue
		idx = row[0]
		npos = idx.index('_')
		i = int(idx[1:npos])-1
		j = int(idx[npos+2:])-1
		index.append((i,j))
	endTime = time.time()
	print('finish reading  submission data', int(endTime-startTime), 's')
	return index

# read in submission indices and return a boolean mask
def readInSubmission2(inPath):
	print('start reading submission data')
	startTime = time.time()
	mask = np.zeros((Globals.nUsers,Globals.nItems),dtype=bool)
	csvReader = csv.reader(open(inPath,encoding='utf-8'))
	abort = True
	for row in csvReader:
		if abort:
			abort = False
			continue
		idx = row[0]
		npos = idx.index('_')
		i = int(idx[1:npos])-1
		j = int(idx[npos+2:])-1
		mask[i,j] = True
	endTime = time.time()
	print('finish reading  submission data', int(endTime-startTime), 's')
	return mask