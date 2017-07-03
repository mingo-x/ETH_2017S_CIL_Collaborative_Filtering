# DESCRIPTION: This file implements the k-means method. Users are classified by k clusters and the distance is defined by the sum of the square distance of observed ratings. Before the clustering, users' mean ratings are subtracted from the observed ratings. A prediction of user i and item j is given by the corresponding value of the center of the cluster user i belongs to, plus his mean ratings. The final predictor is the mean of 11 k-means models, with k varying from 4 to 24 (step = 2).

# USAGE: To train one k-means model, run "python3 code/KMeans.py -k=K" and "python3 code/KMeans.py -k=K -d=1", with K = 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24. "-k" specifies how many clusters are used and "-d" chooses the data split. To combine the k-means models with different ks, run "python3 code/KMeans.py -p=c" and "python3 code/KMeans.py -p=c -d=1". "-p=c" sets the program to do the combination.

import numpy as np
import Initialization
import Globals
import random
import time
import SVD

def kmeans(inData,test,k):
	suffix = '_fixed'+Globals.dataIdx+'.npy'
	if not Globals.fixed:
		suffix = '.npy'
	data = inData.copy()
	nObs = np.count_nonzero(data)
	known = data!=0
	missing = known == False
	uMean = np.empty(Globals.nUsers)
	# substract users' mean ratings
	for i in range(Globals.nUsers):
		uMean[i] = np.mean(data[i])
		data[i] -= uMean[i]
	data[missing] = 0
	# read in previous trained model
	if Globals.warmStart:
		center = np.load('./log/KMeans_center_'+str(k)+suffix)
	# random initialization
	else:
		center = np.empty((k,Globals.nItems))
		for i in range(k):
			idx = random.randint(0,Globals.nUsers-1)
			center[i] = data[idx]
	prev = 1e10
	curr = 1e9

	print('start kmeans')
	startTime = time.time()
	t = 0
	# terminate if the change of error is less than 1e-8
	while np.abs(prev-curr)>1e-8:
		prev = curr
		# new assignment
		assignment = [[] for i in range(k)]
		curr = 0
		for i in range(Globals.nUsers):
			minDist = 1e10
			aidx = 0
			for j in range(k):
				dist = np.sum(np.square((data[i]-center[j])[known[i]]))
				if dist < minDist:
					minDist = dist
					aidx = j
			assignment[aidx].append(i)
			curr += minDist
		curr = np.sqrt(curr/nObs)

		# new center
		for i in range(k):
			if len(assignment[i])!= 0:
				for j in range(Globals.nItems):
					c = np.count_nonzero(known[assignment[i],j])
					if c!=0:
						center[i][j] = np.sum(data[assignment[i],j])/c
					else:
						center[i][j] = 0

		print('t =',t,'rmse =',curr)
		if t%100 == 0:
			np.save('./log/KMeans_center_'+str(k)+suffix,center)
			print('auto save')
		t += 1

	endTime = time.time()
	print('finish kmeans ', int(endTime-startTime), 's')

	A = np.empty((Globals.nUsers,Globals.nItems))
	for i in range(k):
		A[assignment[i]] = center[i]+uMean[i]
	score = SVD.evaluation2(A,test)
	print('test error =',score)

	return A

# combine models with different k
def predictionWithCombi(data,test):
	suffix = '_fixed'+Globals.dataIdx+'.npy'
	if not Globals.fixed:
		suffix = '.npy'
	known = data!=0
	A = np.zeros((Globals.nUsers,Globals.nItems))
	for k in range(4,25,2):
		print('read k =', k,)
		A1 = np.load('./log/Kmeans_A_'+str(k)+suffix)
		print(A.shape)
		A += A1
	A /= 10.0
	score = SVD.evaluation2(A,test)
	print('after combination score =',score)
	# clipping
	# over 5
	mask = A>5
	A[mask] = 5
	# below 1
	mask = A<1
	A[mask] = 1
	score = SVD.evaluation2(A,test)
	print('after clipping score =',score)
	return A

if __name__ == "__main__":
	Initialization.initialization()
	data, test = Initialization.readInData2(idx = Globals.dataIdx)
	# combine
	if Globals.predict=='c':
		A = predictionWithCombi(data,test)
		np.save('./log/Kmeans_A_combi_fixed'+Globals.dataIdx+'.npy',A)
	# evaluate
	elif Globals.predict == 'e':
		A = np.load('./log/Kmeans_A_combi_fixed'+Globals.dataIdx+Globals.modelIdx+'.npy')
		score = SVD.evaluation2(A,test)
		print('score =', score)
	# single k-means
	else:
		A = kmeans(data,test,Globals.k)
		np.save('./log/Kmeans_A_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',A)