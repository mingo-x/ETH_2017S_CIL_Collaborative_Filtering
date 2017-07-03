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
	if len(s) < k:
		k = len(s)
	peers = candidate[np.argpartition(-s,k)[:k]]
	return peers

def predict(u,j,known,score,data):
	peers = peer(u,j,known,score,Globals.k)
	# print(peers)
	pred = 0
	term = 0
	for i in peers:
		# print(u,i,j)
		pred += score[u,i]*data[i,j]
		term += np.abs(score[u,i])
	pred /= term
	pred += mu[u]
		
def output(test,known,score,data):
	print('start predicting')
	A = np.empty((Globals.nUsers,Globals.nItems))
	for i in range(Globals.nUsers):
		if i%100==0:
			print('user', i)
		for j in range(Globals.nItems):
			A[i,j] = predict(i,j,known,score,data)

	e = SVD.evaluation2(A,test)
	print("score =", e)
	mask = A>5
	A[mask] = 5
	mask = A<1
	A[mask] = 1
	e = SVD.evaluation2(A,test)
	print("clipped score =", e)
	np.save('./log/UB_A_'+str(Globals.k)+'.npy')
	print('finish predicting')

if __name__ == '__main__':
	Initialization.initialization()
	data, test = Initialization.readInData2(idx=Globals.dataIdx)
	if Globals.warmStart:
		data, known, mu, score = initialize(data)
	else:
		data, known, mu = initialize(data)
		score = sim(known,data)
	output(test,known,score,data)