import numpy as np
import csv
from sys import argv
import time
import Globals

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
			Globals.predict = argv[i][3]
			print('predict =', Globals.predict)
		elif argv[i].startswith('-i='):
			Globals.modelIdx = '_'+argv[i][3:]
			print('model idx =', Globals.modelIdx)


def readInData(inPath):
# read in data
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