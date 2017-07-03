# DESCRIPTION: This file implements a user-based collaborative filtering method. The similarity between users is assessed by the Pearson correlation coefficient. For user i and item j, the top 500 users closest to user i who have rated item j are found and the ratings of item j by these close users are averaged with the weight being the corresponding Pearson coefficient. Before the averaging, the raw ratings are mean-centered in a user-wise fashion and the mean rating of user i is added back to the weighted average to obtain the final prediction.

# USAGE: To train the model, run "python3 code/UserBased.py" and "python3 code/UserBased.py -d=1". "-d" chooses the training/validation data split.

import Initialization
import Globals
import numpy as np
import SVD
import time

def initialize(data):
	known = data!=0
	mu = [np.mean(data[i,known[i]]) for i in range(Globals.nUsers)]
	# mean-centered user-wisely
	for i in range(Globals.nUsers):
		data[i] -= mu[i]
	# load previously trained model
	if Globals.warmStart:
		score = np.load('./log/UB_sim'+Globals.dataIdx+'.npy')
		return data, known, mu, score
	else:
		return data, known, mu

# Pearson correlation coefficient
def pearson(Is,ru,rv):
	sim = np.dot(ru[Is],rv[Is])/(np.sqrt(np.dot(ru[Is],ru[Is])*np.dot(rv[Is],rv[Is])))
	return sim

# calculate user-user similarity
def sim(known,data):
	print('start calculating similarity')
	index = np.array([i for i in range(Globals.nItems)])
	I = [index[known[i]] for i in range(Globals.nUsers)]
	print(len(I),len(I[0]))
	score = np.empty((Globals.nUsers,Globals.nUsers))
	startTime = time.time()
	for i in range(Globals.nUsers):
		if i%100==0:
			endTime = time.time()
			print('user',i+1,int(endTime-startTime),'s')
			startTime = time.time()
		for j in range(i+1,Globals.nUsers):
			Is = np.intersect1d(I[i],I[j])
			if len(Is)!=0:
				s = pearson(Is,data[i],data[j])
				score[i,j] = s
				score[j,i] = s
	np.save('./log/UB_sim'+Globals.dataIdx+'.npy',score)
	print('finish calculating similarity')
	return score

# find at most k closest users who have rated item j
def peer(u,j,known,score,k):
	#(u,j) should be unobserved
	index = np.array([i for i in range(Globals.nUsers)])
	candidate = index[known[:,j]]
	s = score[u,candidate]
	mask = np.isnan(s)==False
	s = s[mask]
	candidate = candidate[mask]
	if len(s)-1 < k:
		peers = candidate
	else:
		peers = candidate[np.argpartition(-s,k)[:k]]
	return peers

def predict(u,j,known,score,data):
	peers = peer(u,j,known,score,Globals.k)
	pred = 0
	term = 0
	for i in peers:
		pred += score[u,i]*data[i,j]
		term += np.abs(score[u,i])
	pred /= term
	pred += mu[u]
	return pred

def output(test,known,score,data,vali):
	print('start predicting')
	A = np.zeros((Globals.nUsers,Globals.nItems))
	startTime = time.time()
	c = 0
	for i,j in test:
		if c%10000==0:
			endTime = time.time()
			print('user', c, int(endTime-startTime),)
			if Globals.predict == 'v' or Globals.predict == 'k':
				e = SVD.evaluation2(A,vali)
				print('s score =', e)
			startTime = time.time()
		A[i,j] = predict(i,j,known,score,data)
		c += 1

	e = SVD.evaluation2(A,vali)
	print("score =", e)
	# clipping
	mask = A>5
	A[mask] = 5
	mask = A<1
	A[mask] = 1
	e = SVD.evaluation2(A,vali)
	print("clipped score =", e)
	np.save('./log/UB_A_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',A)
	print('finish predicting')

# choose the best k
def chooseK(test,known,score,data,vali):
	for k in range(1000,99,-100):
		print("k =", k)
		Globals.k = k
		output(test,known,score,data,vali)		

if __name__ == '__main__':
	Initialization.initialization()
	data, vali = Initialization.readInData2(idx=Globals.dataIdx)
	if Globals.warmStart:
		data, known, mu, score = initialize(data)
	else:
		data, known, mu = initialize(data)
		score = sim(known,data)
	if Globals.predict == 'v' or Globals.predict == 'k':
		test = []
	else:
		test = Initialization.readInSubmission('./data/sampleSubmission.csv')
	# load (i,j) pairs needed to be predicted
	for i in range(Globals.nUsers):
		for j in range(Globals.nItems):
			if vali[i,j]!=0:
				test.append((i,j))
	# choose best k
	if Globals.predict == 'k':
		chooseK(test,known,score,data,vali)
	# train or validate
	else:
		output(test,known,score,data,vali)