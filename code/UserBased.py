import Initialization
import Globals
import numpy as np
import SVD

def initialize(data):
	known = data!=0
	mu = np.mean(data[known],axis=1)
	data = [data[i]-mu[i] for i in range(Globals.nUsers)]
	if Globals.warmStart:
		score = np.load('./log/UB_sim'+Globals.dataIdx+'.npy')
		return data, known, mu, score
	else:
		return data, known, mu

def pearson(Is,ru,rv):
	sim = np.dot(ru[Is],rv[Is])/(np.sqrt(np.dot(ru[Is],ru[Is]))*np.sqrt(np.dot(rv[IS],rv[Is])))
	return sim

def sim(known,data):
	print('start calculating similarity')
	index = [i for i in range(Globals.nItems)]
	I = [index[known[i]] for i in range(Globals.nUsers)]
	score = np.empty((Globals.nUsers,Globals.nUsers))
	for i in range(Globals.nUsers):
		for j in range(i+1,Globals.nUsers):
			Is = np.intersect1d(I[i],I[j])
			if len(Is)!=0:
				s = pearson(Is,data[i],data[j])
				score[i,j] = s
				score[j,i] = s
	np.save('./log/UB_'+str(Globals.k)+'_sim'+Globals.dataIdx+'.npy')
	print('finish calculating similarity')
	return score

def peer(u,j,known,score):
	#(u,j) should be unobserved
	index = [i for i in range(Globals.nItems)]
	candidate = index[known[:,j]]
	s = score[u,candidate]
	if len(s) < Globals.k:
		k = len(s)
	peers = candidate[np.argpartition(-s,Globals.k)[:Globals.k]]
	return peers

def predict(u,j,known,score,data):
	peers = peer(u,j,known,score)
	pred = 0
	term = 0
	for i in peers:
		pred += score[u,i]*data[j]
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