# current best k = 20. can try with smaller ks
# add clipping?
import Initialization
import SVD
import numpy as np

if __name__ == "__main__":
	Initialization.initialization()
	global outputIdx
	print(outputIdx)
	data = Initialization.readInData('./data/data_train.csv')
	SVD.fillInMissing(data)
	U, S, Vt = SVD.SVD(data)
	global k
	Ak = SVD.predictionWithClipping(U, S, Vt, k=20)
	np.save('./log/PSVD_A20_clip.npy',Ak)
	SVD.writeOutData(Ak)