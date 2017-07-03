# DESCRIPTION: This file implemented a biased l2-norm regularized SVD model with full gradient descent. A bias ci is added for each user i and dj for each item j. The prediction function is U[i]*V[j].T+c[i]+d[j] for user i and item j. Training terminates when the validation error stops decreasing.

# USAGE: To train the model, run "python3 code/RSVDFull2.py -k=32 -l=0.1 -l2=0.1" and "python3 code/RSVDFull2.py -k=32 -l=0.1 -l2=0.1 -d=1". "-k" specifies the number of dimensions used for dimension reduction in SVD, "-l" sets the initial learning rate, "-l2" sets the regularization parameter and "-d" chooses the training/validation data split.

import numpy as np
import math, random, sys
import Globals
import Initialization
import time
import SVD

class RecommenderSystem:
	n_row = 0
	n_col = 0

	data = {} # key: (row, col), val: score
	train_row2col = None
	train_col2row = None
	train_pair = []
	valid_pair = []

	n_train = 0
	n_valid = 0

	U = None
	V = None
	BU = None
	BV = None

	lrate = None
	mu = None
	K = None

	def __init__(self, valid_per_row = 6):
		self.valid_per_row = valid_per_row

	def readData(self, trainPath='./data/train',testPath='./data/test'):
		train = np.load(trainPath+str(Globals.dataIdx)+'.npy')
		test = np.load(testPath+str(Globals.dataIdx)+'.npy')
		self.n_row = Globals.nUsers
		self.n_col = Globals.nItems
		self.train_row2col = [[] for i in range(self.n_row)]
		self.train_col2row = [[] for j in range(self.n_col)]
		for r in range(Globals.nUsers):
			for c in range(Globals.nItems):
				if train[r,c]!=0:
					self.data[(r,c)] = train[r,c]
					self.train_pair.append((r, c))
					self.train_row2col[r].append(c)
					self.train_col2row[c].append(r)
				elif test[r,c]!=0:
					self.data[(r,c)] = test[r,c]
					self.valid_pair.append((r, c))
		row2col = [[] for i in range(self.n_row)]
		for item in self.data:
			r, c = item
			row2col[r].append(c)
		
		print(sum([len(item) for item in self.train_row2col]))
		print(sum([len(item) for item in self.train_col2row]))
		self.n_train = len(self.train_pair)
		print(len(self.train_pair))
		self.n_valid = len(self.valid_pair)
		print(len(self.valid_pair))
		print(len(self.train_pair) + len(self.valid_pair))
		return test

	def initParameters(self, K = 16, lrate = 400, mu = 0.02):
		self.lrate = lrate
		self.mu = mu
		self.K = K
		if Globals.warmStart:
			print('warm start')
			self.U = np.load('./log/RSVDF2_U_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy')
			self.V = np.load('./log/RSVDF2_V_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy')
			self.BU = np.load('./log/RSVDF2_BU_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy')
			self.BV = np.load('./log/RSVDF2_BV_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy')
		else:
			self.U = np.random.rand(self.n_row, self.K)
			self.V = np.random.rand(self.K, self.n_col)
			self.BU = np.random.rand(self.n_row)
			self.BV = np.random.rand(self.n_col)

	# calculate the gradient for one row
	def oneRowGrad(self, r):
		cols = self.train_row2col[r]
		grad = np.zeros(self.K)
		bias_grad = 0
		for c in cols:
			x = np.dot(self.U[r], self.V[:, c])+ self.BU[r] + self.BV[c]
			eff = x  - self.data[(r, c)]
			grad = grad + eff * self.V[:, c]
			bias_grad = bias_grad + eff
		grad = grad / len(cols)
		grad = grad + self.mu * self.U[r]
		bias_grad = bias_grad / len(cols)
		return grad, bias_grad

	# calculate the gradient for one column
	def oneColGrad(self, c):
		rows = self.train_col2row[c]
		grad = np.zeros(self.K)
		bias_grad = 0
		for r in rows:
			x = np.dot(self.U[r], self.V[:, c])+ self.BU[r] + self.BV[c]
			eff = x - self.data[(r, c)]
			grad = grad + eff * self.U[r]
			bias_grad = bias_grad + eff
		grad = grad / len(rows)
		grad = grad + self.mu * self.V[:, c]
		bias_grad = bias_grad / len(rows)
		return grad, bias_grad

	# calculate the row gradients
	def rowGrad(self):
		Ugrad = np.zeros((self.n_row, self.K))
		BUgrad = np.zeros(self.n_row)
		for r in range(self.n_row):
			Ugrad[r], BUgrad[r] = self.oneRowGrad(r)
		return Ugrad, BUgrad

	# calculate the column gradients
	def colGrad(self):
		Vgrad = np.zeros((self.K, self.n_col))
		BVgrad = np.zeros(self.n_col)
		for c in range(self.n_col):
			Vgrad[:, c], BVgrad[c] = self.oneColGrad(c)
		return Vgrad, BVgrad

	def predict(self, r, c):
		x = np.dot(self.U[r], self.V[:, c])+ self.BU[r] + self.BV[c]
		return x

	# calculate the trainind error and the validation error
	def getError(self):
		train_err = 0
		valid_err = 0
		for r, c in self.train_pair:
			train_err += (self.predict(r, c) - self.data[(r, c)]) ** 2
		for r, c in self.valid_pair:
			valid_err += (self.predict(r, c) - self.data[(r, c)]) ** 2
		train_err /= len(self.train_pair)
		valid_err /= len(self.valid_pair)
		return np.sqrt(train_err), np.sqrt(valid_err)

	def train(self):
		print("Start training ...")
		preErr = 1e9
		preTest = 1e9
		i = 0
		while True:
			startTime = time.time()
			Ugrad, BUgrad = self.rowGrad()
			Vgrad, BVgrad = self.colGrad()
			self.U = self.U - self.lrate * Ugrad
			self.V = self.V - self.lrate * Vgrad
			self.BU = self.BU - self.lrate * BUgrad
			self.BV = self.BV - self.lrate * BVgrad
			np.save('./log/RSVDF2_U_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',self.U)
			np.save('./log/RSVDF2_V_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',self.V)
			np.save('./log/RSVDF2_BU_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',self.BU)
			np.save('./log/RSVDF2_BV_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',self.BV)
			endTime = time.time()
			trainErr, testErr =  self.getError()
			print(i,trainErr,testErr,int(endTime-startTime),'s')
			if trainErr > preErr or testErr > preTest:
				if self.lrate > 1e-5:
					self.lrate *= 0.1
					preErr = 1e9
					preTest = 1e9
					print('learning rate =',self.lrate)
				else:
					break
			else:
				preErr = trainErr
				preTest = testErr
			i += 1

	# predict
	def pred(self,test):
		A = np.empty((Globals.nUsers,Globals.nItems))
		for r in range(Globals.nUsers):
			for c in range(Globals.nItems):
				A[r,c] = self.predict(r,c)
		# clipping
		# over 5
		mask = A>5
		A[mask] = 5
		# below 1
		mask = A<1
		A[mask] = 1
		print('finish clipping')
		score = SVD.evaluation2(A,test)
		print('after clipping score =',score)
		np.save('./log/RSVDF2_A_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',A)

if __name__ == "__main__":
	Initialization.initialization()
	RS = RecommenderSystem()
	test = RS.readData()
	RS.initParameters(K = Globals.k, lrate = Globals.lrate, mu = Globals.l2)
	RS.train()
	RS.pred(test)
