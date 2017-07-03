# DESCRIPTION: This file implements the SVD method with dimension reduction. The missing values of each movie are filled with the mean of the movie's observed ratings. The number of k is currently set to 12.

# USAGE: To train the model , run "python3 code/PlainSVD.py -k=12" and "python3 code/PlainSVD.py -k=12 -d=1". "-k" sets the number of dimensions used and "-d" chooses the data split.

import Initialization
import SVD
import numpy as np
import Globals

# choose the best number of dimensions
def chooseK(U,S,Vt,test):
	for k in range(4,17,2):
		SVD.predictionWithClipping(U, S, Vt, k, test)

if __name__ == "__main__":
	Initialization.initialization()
	data, test = Initialization.readInData2(idx=Globals.dataIdx)
	# fill in missing data with movies' mean ratings
	SVD.fillInMissing(data)
	# SVD
	U, S, Vt = SVD.SVD(data)
	# choose best k
	if Globals.predict == 'k':
		chooseK(U,S,Vt,test)
	# train and predict
	else:
		Ak = SVD.predictionWithClipping(U, S, Vt, Globals.k,test)
		np.save('./log/PSVD_A_'+str(Globals.k)+'_fixed'+Globals.dataIdx+'.npy',Ak)