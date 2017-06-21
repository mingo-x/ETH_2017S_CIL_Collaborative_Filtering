# continuous rating for prediction
import numpy as np
import csv
from sys import argv

# read in training data
data = np.zeros((10000,1000))
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

# mean for each item
for i in range(1000):
	missing = data[:,i]==0
	known = missing==False
	mean_of_known = np.mean(data[known,i])
	print("mean of col ",i+1,mean_of_known)
	data[missing,i] = mean_of_known

# write prediction
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
	csvWriter.writerow([idx,data[i,j]])
	print(idx,data[i,j])