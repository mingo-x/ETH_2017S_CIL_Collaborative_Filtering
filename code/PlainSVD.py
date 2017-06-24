# current best k = 20. can try with smaller ks
# with clipping 1.00505 without clipping by step 1.00501
import Initialization
import SVD
import numpy as np
import Globals

def chooseK(U,S,Vt,test):
	for k in range(5,26,5):
		SVD.predictionWithClipping(U, S, Vt, k, test)

if __name__ == "__main__":
	Initialization.initialization()
	if Globals.fixed:
		data, test = Initialization.readInData2(idx=Globals.dataIdx)
	else:
		data = Initialization.readInData('./data/data_train.csv')
	SVD.fillInMissing(data)
	U, S, Vt = SVD.SVD(data)
	if Globals.predict == 'k':
		chooseK(U,S,Vt,test)
	else:
		Ak = SVD.predictionWithClipping(U, S, Vt, 20,test)
		if Globals.fixed:
			np.save('./log/PSVD_A_20_fixed'+Globals.dataIdx+'.npy',Ak)
		else:
			np.save('./log/PSVD_A_20.npy',Ak)
		# SVD.writeOutData(Ak)