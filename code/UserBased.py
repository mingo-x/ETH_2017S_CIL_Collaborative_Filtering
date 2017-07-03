import Initialization
import Globals
import numpy as np
import SVD
import time

def initialize(data):
	known = data!=0
	mu = [np.mean(data[i,known[i]]) for i in range(Globals.nUsers)]
	for i in range(Globals.nUsers):
		data[i] -= mu[i]
	if Globals.warmStart:
		score = np.load('./log/UB_sim'+Globals.dataIdx+'.npy')
		return data, known, mu, score
	else:
		return data, known, mu

def pearson(Is,ru,rv):
	sim = np.dot(ru[Is],rv[Is])/(np.sqrt(np.dot(ru[Is],ru[Is])*np.dot(rv[Is],rv[Is])))
	return sim

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
	if np.isnan(pred):
		print(pred,term,peers.shape)
		
def output(test,known,score,data,vali):
	print('start predicting')
	A = np.empty((Globals.nUsers,Globals.nItems))
	startTime = time.time()
	c = 0
	for i,j in test:
		if c%10000==0:
			endTime = time.time()
			print('user', c, int(endTime-startTime), 's')
			startTime = time.time()
		A[i,j] = predict(i,j,known,score,data)
		c += 1

	e = SVD.evaluation2(A,vali)
	print("score =", e)
	mask = A>5
	A[mask] = 5
	mask = A<1
	A[mask] = 1
	e = SVD.evaluation2(A,vali)
	print("clipped score =", e)
	np.save('./log/UB_A_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',A)
	print('finish predicting')

if __name__ == '__main__':
	Initialization.initialization()
	data, vali = Initialization.readInData2(idx=Globals.dataIdx)
	if Globals.warmStart:
		data, known, mu, score = initialize(data)
	else:
		data, known, mu = initialize(data)
		score = sim(known,data)
	test = Initialization.readInSubmission('./data/sampleSubmission.csv')
	for i in range(Globals.nUsers):
		for j in range(Globals.nItems):
			if vali[i,j]!=0:
				test.append((i,j))
	output(test,known,score,data)