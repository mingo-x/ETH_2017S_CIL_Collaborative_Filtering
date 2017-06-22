import numpy as np
import csv
from sys import argv
import time

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