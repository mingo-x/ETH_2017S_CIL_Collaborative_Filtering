# current best k = 20. can try with smaller ks
# add clipping?
import Initialization
import SVD


if __name__ == "__main__":
	Initialization.initialization()
	print('k =',Globals.k)
	print('output idx =',Globals.outputIdx)
	data = Initialization.readInData('./data/data_train.csv')
	SVD.fillInMissing(data)
	U, S, Vt = SVD.SVD(data)
	Ak = SVD.prediction(U, S, Vt,SVD.k)
	SVD.writeOutData(Ak)