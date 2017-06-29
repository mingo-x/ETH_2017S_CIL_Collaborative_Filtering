import numpy as np
import math, random, sys
import Globals
import Initialization

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
	BU = None
	V = None
	BV = None

	lrate = None
	mu = None
	K = None

	def __init__(self, valid_per_row = 6):
		self.valid_per_row = valid_per_row

	def readData(self, filename):
		f = open(filename, 'r')
		lines = f.readlines()[1: ]
		print(len(lines))
		for line in lines:
			nums = line.replace('r', '').replace('_c', ',').split(',')
			r, c, s = [int(num) for num in nums]
			self.data[(r - 1, c - 1)] = s
			self.n_row = max(self.n_row, r)
			self.n_col = max(self.n_col, c)
		row2col = [[] for i in range(self.n_row)]
		for item in self.data:
			r, c = item
			s = self.data[item]
			row2col[r].append(c)
		self.train_row2col = [[] for i in range(self.n_row)]
		self.train_col2row = [[] for j in range(self.n_col)]
		for r, item in enumerate(row2col):
			num = int(self.valid_per_row / 100.0 * len(item))
			order = [i for i in range(len(item))]
			random.shuffle(order)
			idx = order[0: num]
			for i, c in enumerate(item):
				if i in idx:
					self.valid_pair.append((r, c))
				else:
					self.train_pair.append((r, c))
					self.train_row2col[r].append(c)
					self.train_col2row[c].append(r)
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
		self.U = np.random.rand(self.n_row, self.K)
		self.BU = np.random.rand(self.n_row)
		self.V = np.random.rand(self.K, self.n_col)
		self.BV = np.random.rand(self.n_col)

	def oneRowGrad(self, r, Unorm):
		cols = self.train_row2col[r]
		grad = np.zeros(self.K)
		bias_grad = 0
		for c in cols:
			x = sigmoid(np.dot(self.U[r], self.V[:, c]) + self.BU[r] + self.BV[c])
			eff = 2 * (4 * x + 1 - self.data[(r, c)]) * 4 * x * (1 - x)
			grad = grad + eff * self.V[:, c]
			bias_grad = bias_grad + eff
		grad = grad / self.n_train
		grad = grad + self.mu / Unorm * self.U[r]
		bias_grad = bias_grad / self.n_train
		return grad, bias_grad

	def oneColGrad(self, c, Vnorm):
		rows = self.train_col2row[c]
		grad = np.zeros(self.K)
		bias_grad = 0
		for r in rows:
			x = sigmoid(np.dot(self.U[r], self.V[:, c]) + self.BU[r] + self.BV[c])
			eff = 2 * (4 * x + 1 - self.data[(r, c)]) * 4 * x * (1 - x)
			grad = grad + eff * self.U[r]
			bias_grad = bias_grad + eff
		grad = grad / self.n_train
		grad = grad + self.mu / Vnorm * self.V[:, c]
		bias_grad = bias_grad / self.n_train
		return grad, bias_grad

	def rowGrad(self):
		Unorm = 0
		for i in range(self.n_row):
			for j in range(self.K):
				Unorm += self.U[i, j] ** 2
		Unorm = np.sqrt(Unorm)
		Ugrad = np.zeros((self.n_row, self.K))
		BUgrad = np.zeros(self.n_row)
		for r in range(self.n_row):
			Ugrad[r], BUgrad[r] = self.oneRowGrad(r, Unorm)
		return Ugrad, BUgrad

	def colGrad(self):
		Vnorm = 0
		for i in range(self.K):
			for j in range(self.n_col):
				Vnorm += self.V[i, j] ** 2
		Vnorm = np.sqrt(Vnorm)
		Vgrad = np.zeros((self.K, self.n_col))
		BVgrad = np.zeros(self.n_col)
		for c in range(self.n_col):
			Vgrad[:, c], BVgrad[c] = self.oneColGrad(c, Vnorm)
		return Vgrad, BVgrad

	def predict(self, r, c):
		x = sigmoid(np.dot(self.U[r], self.V[:, c]) + self.BU[r] + self.BV[c])
		return 4 * x + 1

	def getError(self):
		train_err = 0
		valid_err = 0
		for r, c in self.train_pair:
			train_err += (self.predict(r, c) - self.data[(r, c)]) ** 2
		for r, c in self.valid_pair:
			valid_err += (self.predict(r, c) - self.data[(r, c)]) ** 2
		train_err /= len(self.train_pair)
		valid_err /= len(self.valid_pair)
		return train_err, valid_err

	def train(self, n_step = 200):
		print("Start training ...")
		for i in range(n_step):
			Ugrad, BUgrad = self.rowGrad()
			Vgrad, BVgrad = self.colGrad()
			self.U = self.U - self.lrate * Ugrad
			self.BU = self.BU - self.lrate * BUgrad
			self.V = self.V - self.lrate * Vgrad
			self.BV = self.BV - self.lrate * BVgrad
			print(i, self.getError())

	def writeSubmissionFile(self, filename):
		print("Start prediction ...")
		write_file = filename.replace(".csv", "-" + str(self.K) + "-" + str(self.lrate) + "-" + str(self.mu) + ".csv")
		f = open(filename, 'r')
		lines = f.readlines()[1: ]
		print(len(lines))
		f.close()
		f = open(write_file, 'w')
		f.write("Id,Prediction\n")
		for line in lines:
			nums = line.replace('r', '').replace('_c', ',').split(',')
			r, c, s = [int(num) for num in nums]
			s = self.predict(r - 1, c - 1)
			f.write("r" + str(r) + "_c" + str(c) + "," + str(s) + "\n")
		f.close()

	def pred(self):
		A = np.empty((Globals.nUsers,Globals.nItems))
		for r in range(Globals.nUsers):
			for c in range(Globals.nItems):
				A[r,c] = self.predict(r,c)

		np.save('./log/GRSVD_A_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',A)


if __name__ == "__main__":
	Initialization.initialization()
	random.seed(0)
	np.random.seed(0)
	RS = RecommenderSystem()
	RS.readData("./data/data_train.csv")
	RS.initParameters(K = Globals.k, lrate = Globals.lrate, mu = 0.02)
	RS.train()
	RS.pred()
	# RS.writeSubmissionFile("submission.csv")
