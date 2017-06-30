import numpy as np
import math, random, sys
import Globals
import Initialization
import time

def sigmoid(x):
	return math.exp(-np.logaddexp(0, -x))

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

	def initParameters(self, K = 16, lrate = 400, mu = 0.02):
		self.lrate = lrate
		self.mu = mu
		self.K = K
		if Globals.warmStart:
			print('warm start')
			self.U = np.load('./log/RSVDF_U_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy')
			self.V = np.load('./log/RSVDF_V_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy')
		else:
			self.U = np.random.rand(self.n_row, self.K)
			self.V = np.random.rand(self.K, self.n_col)

	def oneRowGrad(self, r, Unorm):
		cols = self.train_row2col[r]
		grad = np.zeros(self.K)
		for c in cols:
			x = np.dot(self.U[r], self.V[:, c])
			eff = x  - self.data[(r, c)]
			grad = grad + eff * self.V[:, c]
		grad = grad / self.n_train
		grad = grad + self.mu / Unorm * self.U[r]
		return grad

	def oneColGrad(self, c, Vnorm):
		rows = self.train_col2row[c]
		grad = np.zeros(self.K)
		for r in rows:
			x = np.dot(self.U[r], self.V[:, c])
			eff = x - self.data[(r, c)]
			grad = grad + eff * self.U[r]
		grad = grad / self.n_train
		grad = grad + self.mu / Vnorm * self.V[:, c]
		return grad

	def rowGrad(self):
		Unorm = 0
		for i in range(self.n_row):
			for j in range(self.K):
				Unorm += self.U[i, j] ** 2
		Unorm = np.sqrt(Unorm)
		Ugrad = np.zeros((self.n_row, self.K))
		for r in range(self.n_row):
			Ugrad[r] = self.oneRowGrad(r,Unorm)
		return Ugrad

	def colGrad(self):
		Vnorm = 0
		for i in range(self.K):
			for j in range(self.n_col):
				Vnorm += self.V[i, j] ** 2
		Vnorm = np.sqrt(Vnorm)
		Vgrad = np.zeros((self.K, self.n_col))
		for c in range(self.n_col):
			Vgrad[:, c] = self.oneColGrad(c,Vnorm)
		return Vgrad

	def predict(self, r, c):
		x = np.dot(self.U[r], self.V[:, c])
		return x

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
		prevErr = 1e10
		i = 0
		while True:
			startTime = time.time()
			Ugrad = self.rowGrad()
			Vgrad = self.colGrad()
			self.U = self.U - self.lrate * Ugrad
			self.V = self.V - self.lrate * Vgrad
			np.save('./log/RSVDF_U_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',self.U)
			np.save('./log/RSVDF_V_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',self.V)
			endTime = time.time()
			trainErr, testErr =  self.getError()
			print(i,trainErr,testErr,int(endTime-startTime),'s')
			if testErr > prevErr:
				if self.lrate > 1e-5:
					self.lrate *= 0.1
					print('learning rate =',self.lrate)
				else:
					break
			prevErr = testErr
			i += 1

	def pred(self):
		A = np.empty((Globals.nUsers,Globals.nItems))
		for r in range(Globals.nUsers):
			for c in range(Globals.nItems):
				A[r,c] = self.predict(r,c)

		np.save('./log/RSVDF_A_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',A)


if __name__ == "__main__":
	Initialization.initialization()
	RS = RecommenderSystem()
	RS.readData()
	RS.initParameters(K = Globals.k, lrate = Globals.lrate, mu = 0.02)
	RS.train()
	RS.pred()
	# RS.writeSubmissionFile("submission.csv")
