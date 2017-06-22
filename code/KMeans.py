# mean prediction of ensemble of 10 runs
# of K-means with K ranging from 4 to 24
import numpy as np
import Initialization
import Globals
import random
import time
import SVD

def kmeans(inData,k):
	data = inData.copy()
	nObs = np.count_nonzero(data)
	known = data!=0
	uMean = np.empty(Globals.nUsers)
	for i in range(Globals.nUsers):
		uMean[i] = np.mean(data[i])
		data[i] -= uMean[i]
	center = np.empty((k,Globals.nItems))
	for i in range(k):
		center[i] = random.normalvariate(0,1)
		# idx = random.randint(0,Globals.nUsers-1)
		# center[i] = data[idx]
	prev = 1e10
	curr = 1e9

	print('start kmeans')
	startTime = time.time()
	t = 0
	while prev-curr>1e-7:
		prev = curr
		# assign
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
		# print(len(assignment[0]),len(assignment[1]),len(assignment[2]),len(assignment[3]))

		# mean
		for i in range(k):
			print(len(assignment[i]),)
			if len(assignment[i])!= 0:
				center[i] = np.mean(data[assignment[i]],axis=0)
		# print(center[0],center[1])

		# if t%1000 == 0:
		print('t =',t,'rmse =',curr)
		if t%10000 == 0:
			np.save('./log/KMeans_center_'+str(k)+'.npy',center)
			print('auto save')
		t += 1

	print('t =',t,'rmse =', curr)
	endTime = time.time()
	print('finish kmeans ', int(endTime-startTime), 's')

	A = np.empty((Globals.nUsers,Globals.nItems))
	for i in range(k):
		A[assignment[i]] = center[i]+uMean[i]

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
	data = Initialization.readInData('./data/data_train.csv')
	A = kmeans(data,Globals.k)
	np.save('./log/Kmeans_A_'+str(Globals.k)+Globals.modelIdx+'.npy',A)
	SVD.writeOutData(A)