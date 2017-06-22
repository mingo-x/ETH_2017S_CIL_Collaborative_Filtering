# mean prediction of ensemble of 10 runs
# of K-means with K ranging from 4 to 24
import numpy as np
import Initialization
import Globals
import random
import time
import SVD

def kmeans(data,k):
	nObs = np.count_nonzero(data)
	newCenter = np.empty((k,Globals.nItems))
	for i in range(k):
		idx = random.randint(0,Globals.nUsers-1)
		newCenter[i] = data[idx]
	dist = 1e10

	print('start kmeans')
	startTime = time.time()
	t = 0
	while dist>1e-6:
		center = newCenter.copy()
		# assign
		assignment = [[]]*k
		sumMinDist = 0
		for i in range(Globals.nUsers):
			minDist = 1e10
			known = data[i]!=0
			aidx = 0
			for j in range(k):
				dist = np.sum(np.square((data[i]-center[j])[known]))
				if dist < minDist:
					minDist = dist
					aidx = j
			assignment[aidx].append(i)
			sumMinDist += minDist
		print(assignment)
		# mean
		for i in range(k):
			newCenter[i] = np.mean(data[assignment[i]],axis=0)

		dist = np.linalg.norm(newCenter-center)
		if t%5000 == 0:
			RMSE = np.sqrt(sumMinDist/nObs)
			print('t =',t,'rmse =',RMSE, 'dist =', dist)
		if t%100000 == 0:
			np.save('./log/KMeans_center_'+str(k)+'.npy',center)
			print('auto save')
		
		t += 1
	RMSE = np.sqrt(sumMinDist/nObs)
	print('t =',t,'rmse =',RMSE, 'dist =', dist)
	endTime = time.time()
	print('finish kmeans ', int(endTime-startTime), 's')

	center = newCenter.copy()
	A = np.empty((Globals.nUsers,Globals.nItems))
	for i in range(k):
		A[assignment[i]] = center[i]

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