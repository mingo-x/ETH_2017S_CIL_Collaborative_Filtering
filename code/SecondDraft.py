# continuous rating for prediction
import numpy as np
import csv
from sys import argv
import time

# read in training data
data = np.zeros((10000,1000))
print('start reading data')
csvReader = csv.reader(open('./data/data_train.csv',encoding='utf-8'))
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
print('finish reading data')

# fill in missing data
print('start filling in missing data')
for i in range(1000):
	missing = data[:,i]==0
	known = missing==False
	mean_of_known = np.mean(data[known,i])
	# print("mean of col ",i+1,mean_of_known)
	data[missing,i] = mean_of_known
print('finish filling in missing data')

# SVD
print('start SVD')
startTime = time.time()
U, s, Vt = np.linalg.svd(data, full_matrices=True)
endTime = time.time()
print('finish SVD', U.shape, s.shape, Vt.shape, int(endTime-startTime), 's')
S = np.zeros((10000, 1000))
S[:1000, :1000] = np.diag(s)
A = U.dot(S).dot(Vt)
# whether two arrays are element-wise equal within a tolerance
print('close?', np.allclose(data, A))
k = 1000
Sk = S.copy()
Sk[k:, k:] = 0
Ak = U.dot(Sk).dot(Vt)

# write prediction
print('start writing data')
csvReader = csv.reader(open('./data/sampleSubmission.csv',encoding='utf-8'))
idx = ''
if len(argv) > 1 :
	idx = argv[1]
csvWriter = csv.writer(open('./data/prediction'+idx+'.csv','w',newline=''))
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
	# print(idx,data[i,j])