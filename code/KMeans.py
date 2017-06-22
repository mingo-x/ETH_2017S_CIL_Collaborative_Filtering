# mean prediction of ensemble of 10 runs
# of K-means with K ranging from 4 to 24、
import numpy as np
import Initialization
import Globals
import random
import time

def kmeans(data,k):
	center = np.empty((k,Globals.nItems))
	assignment = [[]]*k
	newCenter = np.zeros((k,Globals.nItems))
	for i in range(k):
		idx = random.randint(0,Globals.nUsers-1)
		center[i] = data[idx,:]

	t = 0
	while dist>1e-6:
		center = newCenter.copy()
		# assign
		for i in range(Globals.nUsers):
			minDist = 1e10
			known = data[i]!=0
			aidx = 0
			for j in range(k):
				dist = np.sum(np.square((data[i]-center[j])[known]))
				if dist < minDist:
					minDist = dist
					aidx = j
			assignment[j].append(i)

		# mean
		for i in range(k):
			newCenter[i] = np.mean(data[assignment[i]],axis=0)

		dist = np.linalg.norm(newCenter-center)
		if t%5000 == 0:
			print('t =',t,'dist =',dist)
		t += 1

	center = newCenter.copy()


if __name__ == "__main__":
