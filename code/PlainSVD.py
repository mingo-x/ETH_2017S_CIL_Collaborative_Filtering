# current best k = 20. can try with smaller ks
# with clipping 1.00505 without clipping by step 1.00501
import Initialization
import SVD
import numpy as np
import Globals

if __name__ == "__main__":
	Initialization.initialization()
	data = Initialization.readInData('./data/data_train.csv')
	SVD.fillInMissing(data)
	U, S, Vt = SVD.SVD(data)
	Ak = SVD.predictionWithClipping(U, S, Vt, k=20)
	np.save('./log/PSVD_A20_clip.npy',Ak)
	SVD.writeOutData(Ak)