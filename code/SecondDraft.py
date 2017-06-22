import SVD
import numpy as np

if __name__ == "__main__":
	SVD.initialization()
	print('k =',SVD.k)
	print('output idx =',SVD.outputIdx)
	data = SVD.readInData('./data/data_train.csv')
	SVD.fillInMissing(data)
	U, S, Vt = SVD.SVD(data)
	Ak = SVD.prediction(U, S, Vt)
	SVD.writeOutData()