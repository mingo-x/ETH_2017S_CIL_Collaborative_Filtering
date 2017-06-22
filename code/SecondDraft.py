import SVD
import numpy as np

if __name__ == "__main__":
	SVD.initialization()
	print('k =',SVD.k)
	print('output idx =',SVD.outputIdx)
	data = SVD.readInData('./data/data_train.csv')
	# train, testMask = SVD.splitData(data)
	print('num of data =', np.count_nonzero(data))
	SVD.fillInMissing(data)
	print('num of data =', np.count_nonzero(data))
	U, S, Vt = SVD.SVD(data)
	Ak = SVD.prediction(U, S, Vt)
	# print('RMSE =',SVD.evaluation(data,Ak,testMask))
	SVD.writeOutData()